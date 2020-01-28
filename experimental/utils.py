import cv2
import string
import random
import numpy as np
import trt_pose.coco
from operator import itemgetter
from sklearn.utils.linear_assignment_ import linear_assignment

import person
import config as cfg

topology = trt_pose.coco.coco_category_to_topology(cfg.human_pose)

def id_gen(size=6, chars=string.ascii_uppercase + string.digits):
    '''
    https://pythontips.com/2013/07/28/generating-a-random-string/
    input: id_gen(3, "6793YUIO")
    output: 'Y3U'
    '''
    return ''.join(random.choice(chars) for x in range(size))

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
        if(IOU_mat[m[0], m[1]] < iou_thrd):
            unmatched_trackers.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(m.reshape(1,2))

    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

def source_capture(source):
    source = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(source)

    fourcc_cap = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc_cap)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.h)
    return cap

class img_obj(object):
    def __init__(self, offset=50):
        self.offset = 50
        self.fontScale = 1
        self.thickness = 2
        self.box_color = (195, 195, 89)
        self.text_color = (151, 187, 106)
        self.centroid_color = (223, 183, 190)


    def annotate(self, tracker, image, boxes):
        '''
        Used to return image with
        person instances designated 
        by the bounding box and a 
        marker at the centroid.
        Annotated with tracker id
        and activity label
        '''
        for row in topology:
            try:
                a_idx, b_idx = row[2:]
                a_part, b_part = cfg.body_dict[int(a_idx.data.cpu().numpy())], cfg.body_dict[int(b_idx.data.cpu().numpy())]
                a_coord, b_coord = tracker.pose_dict[a_part], tracker.pose_dict[b_part]
                cv2.line(image, a_coord, b_coord, tracker.skeleton_color, 2)
            except KeyError:
                pass

        if boxes:
            try:
                x1, y1, x2, y2 = tracker.bbox
                image = cv2.rectangle(image, (x1 - self.offset, y1 - self.offset), 
                                             (x2 + self.offset, y2 + self.offset), 
                                             self.box_color, 2) 
                image = cv2.drawMarker(image, tracker.centroid, self.centroid_color, 0, 30, self.thickness) 
                cv2.putText(image, tracker.id, (x1 - self.offset, y1 - self.offset), \
                                   cv2.FONT_HERSHEY_SIMPLEX, self.fontScale, self.text_color, self.thickness) 
                cv2.putText(image, str(tracker.activity), (x1 - self.offset, y1 - self.offest), \
                                   cv2.FONT_HERSHEY_SIMPLEX, self.fontScale, self.text_color, self.thickness) 
            except:
                pass

        return image

    def get_crop(self, bbox, image):
        '''
        Helper for sampling image crops
        '''
        return image[x1:x2, y1:y2, :]



def update_trackers(trackers, bboxes):
    track_boxes = [tracker.bbox for tracker in trackers]
    matched, unmatched_trackers, unmatched_detections = tracker_match(track_boxes, [b[0] for b in bboxes])

    for idx, jdx in matched:
        trackers[idx].set_bbox(bboxes[jdx][0])
        trackers[idx].set_pose(bboxes[jdx][1])

    for idx in unmatched_detections:
        try:
            trackers[idx].count += 1
            if trackers[idx].count > trackers[idx].expiration:
                trackers.pop(idx)
        except:
            pass

    for idx in unmatched_trackers:
        p = person.PersonTracker()
        p.set_bbox(bboxes[idx][0])
        p.set_pose(bboxes[idx][1])
        trackers.append(p)
    return trackers
