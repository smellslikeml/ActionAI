import os
import sys
import cv2
import csv
import time
import json
import torch
import string
import random
import PIL.Image
import numpy as np
from collections import deque
from operator import itemgetter
from sklearn.utils.linear_assignment_ import linear_assignment

from pprint import pprint

import trt_pose.coco
import trt_pose.models
from torch2trt import TRTModule
import torchvision.transforms as transforms
from trt_pose.parse_objects import ParseObjects

model_w = 224
model_h = 224

ASSET_DIR = 'models/'
OPTIMIZED_MODEL = ASSET_DIR + 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'

body_labels = {0:'nose', 1: 'lEye', 2: 'rEye', 3:'lEar', 4:'rEar', 5:'lShoulder', 6:'rShoulder',
               7:'lElbow', 8:'rElbow', 9:'lWrist', 10:'rWrist', 11:'lHip', 12:'rHip', 13:'lKnee', 14:'rKnee',
              15:'lAnkle', 16:'rAnkle', 17:'neck'}
body_idx = dict([[v,k] for k,v in body_labels.items()])

with open(ASSET_DIR + 'human_pose.json', 'r') as f:
    human_pose = json.load(f)

model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')


def id_gen(size=6, chars=string.ascii_uppercase + string.digits):
    '''
    https://pythontips.com/2013/07/28/generating-a-random-string/
    input: id_gen(3, "6793YUIO")
    output: 'Y3U'
    '''
    return ''.join(random.choice(chars) for x in range(size))

def preprocess(image):
    global device
    device = torch.device('cuda')
    image = cv2.resize(image, (model_h, model_w))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

def inference(image):
    data = preprocess(image)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf) #, cmap_threshold=0.15, link_threshold=0.15)
    body_dict = draw_objects(image, counts, objects, peaks)
    return image, body_dict

def IOU(boxA, boxB):
    # pyimagesearch: determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def get_bbox(kp_list):
    bbox = []
    for aggs in [min, max]:
        for idx in range(2):
            bound = aggs(kp_list, key=itemgetter(idx))[idx]
            bbox.append(bound)
    return bbox

def tracker_match(trackers, detections, iou_thrd = 0.3):
    '''
    From current list of trackers and new detections, output matched detections,
    unmatched trackers, unmatched detections.
    https://towardsdatascience.com/computer-vision-for-tracking-8220759eee85
    '''

    IOU_mat= np.zeros((len(trackers),len(detections)),dtype=np.float32)
    for t,trk in enumerate(trackers):
        for d,det in enumerate(detections):
            IOU_mat[t,d] = IOU(trk,det)

    # Produces matches
    # Solve the maximizing the sum of IOU assignment problem using the
    # Hungarian algorithm (also known as Munkres algorithm)

    matched_idx = linear_assignment(-IOU_mat)

    unmatched_trackers, unmatched_detections = [], []
    for t,trk in enumerate(trackers):
        if(t not in matched_idx[:,0]):
            unmatched_trackers.append(t)

    for d, det in enumerate(detections):
        if(d not in matched_idx[:,1]):
            unmatched_detections.append(d)

    matches = []

    # For creating trackers we consider any detection with an
    # overlap less than iou_thrd to signifiy the existence of
    # an untracked object

    for m in matched_idx:
        if(IOU_mat[m[0],m[1]] < iou_thrd):
            unmatched_trackers.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(m.reshape(1,2))

    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class PersonTracker(object):
    def __init__(self):
        self.id = id_gen() #int(time.time() * 1000)
        self.q = deque(maxlen=10)
        return

    def set_bbox(self, bbox):
        self.bbox = bbox
        x1, y1, x2, y2 = bbox
        self.h = 1e-6 + x2 - x1
        self.w = 1e-6 + y2 - y1
        self.centroid = tuple(map(int, ( x1 + self.h / 2, y1 + self.w / 2)))
        return

    def update_pose(self, pose_dict):
        ft_vec = np.zeros(2 * len(body_labels))
        for ky in pose_dict:
            idx = body_idx[ky]
            ft_vec[2 * idx: 2 * (idx + 1)] = 2 * (np.array(pose_dict[ky]) - np.array(self.centroid)) / np.array((self.h, self.w))
        self.q.append(ft_vec)
        return

    def annotate(self, image):
        x1, y1, x2, y2 = self.bbox
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
        image = cv2.putText(image, self.activity, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        image = cv2.drawMarker(image, self.centroid, (255, 0, 0), 0, 30, 4)
        return image

class DrawObjects(object):

    def __init__(self, topology, body_labels):
        self.topology = topology
        self.body_labels = body_labels

    def __call__(self, image, object_counts, objects, normalized_peaks):
        topology = self.topology
        height = image.shape[0]
        width = image.shape[1]

        K = topology.shape[0]
        count = int(object_counts[0])
        K = topology.shape[0]
        body_list = []
        for i in range(count):
            body_dict = {}
            color = (112,107,222)
            obj = objects[0][i]
            C = obj.shape[0]
            for j in range(C):
                k = int(obj[j])
                if k >= 0:
                    peak = normalized_peaks[0][j][k]
                    x = round(float(peak[1]) * width)
                    y = round(float(peak[0]) * height)
                    cv2.circle(image, (x, y), 3, color, 2)
                    body_dict[self.body_labels[j]] = (x,y)
            body_list.append(body_dict)
            for k in range(K):
                c_a = topology[k][2]
                c_b = topology[k][3]
                if obj[c_a] >= 0 and obj[c_b] >= 0:
                    peak0 = normalized_peaks[0][c_a][obj[c_a]]
                    peak1 = normalized_peaks[0][c_b][obj[c_b]]
                    x0 = round(float(peak0[1]) * width)
                    y0 = round(float(peak0[0]) * height)
                    x1 = round(float(peak1[1]) * width)
                    y1 = round(float(peak1[0]) * height)
                    cv2.line(image, (x0, y0), (x1, y1), color, 2)
        return body_list

topology = trt_pose.coco.coco_category_to_topology(human_pose)
parse_objects = ParseObjects(topology)
draw_objects = DrawObjects(topology, body_labels)

source = sys.argv[1]
source = int(source) if source.isdigit() else source
cap = cv2.VideoCapture(source)

w = int(cap.get(3))
h = int(cap.get(4))

fourcc_cap = cv2.VideoWriter_fourcc(*'MJPG')
cap.set(cv2.CAP_PROP_FOURCC, fourcc_cap)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

DEBUG = True
WRITE2CSV = False
WRITE2VIDEO = True
RUNSECONDARY = True

if WRITE2CSV:
    activity = os.path.basename(source)
    dataFile = open('data/{}.csv'.format(activity),'w')
    newFileWriter = csv.writer(dataFile)

if WRITE2VIDEO:
    # Define the codec and create VideoWriter object
    name = 'out.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(name, fourcc, 30.0, (w, h))

if RUNSECONDARY:
    import tensorflow as tf
    secondary_model = tf.keras.models.load_model('models/lstm_spin_squat.h5')
    window = 3
    pose_vec_dim = 36
    motion_dict = {0: 'spin', 1: 'squat'}

trackers = []
while True:

    ret, frame = cap.read()
    bboxes = []
    if ret:

        image, pose_list = inference(frame)
        for body in pose_list:
            bbox = get_bbox(list(body.values()))
            bboxes.append((bbox, body))

        track_boxes = [tracker.bbox for tracker in trackers]
        matched, unmatched_trackers, unmatched_detections = tracker_match(track_boxes, [b[0] for b in bboxes])

        for idx, jdx in matched:
            trackers[idx].set_bbox(bboxes[jdx][0])
            trackers[idx].update_pose(bboxes[jdx][1])

        for idx in unmatched_detections:
            try:
                trackers.pop(idx)
            except:
                pass

        for idx in unmatched_trackers:
            person = PersonTracker()
            person.set_bbox(bboxes[idx][0])
            person.update_pose(bboxes[idx][1])
            trackers.append(person)

        if RUNSECONDARY:
            for tracker in trackers:
                print(len(tracker.q))
                if len(tracker.q) >= 3:
                    sample = np.array(list(tracker.q)[:3])
                    sample = sample.reshape(1, pose_vec_dim, window)
                    pred_activity = motion_dict[np.argmax(secondary_model.predict(sample)[0])]
                    tracker.activity = pred_activity
                    image = tracker.annotate(image)
                    print(pred_activity)

        if DEBUG:
            pprint([(tracker.id, np.vstack(tracker.q)) for tracker in trackers])

        if WRITE2CSV:
            for tracker in trackers:
                print(len(tracker.q))
                if len(tracker.q) >= 3:
                    newFileWriter.writerow([activity] + list(np.hstack(list(tracker.q)[:3])))

        if WRITE2VIDEO:
            out.write(image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()

try:
    dataFile.close()
except:
    pass

try:
    out.release()
except:
    pass
