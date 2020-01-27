import os
import PIL
import cv2
import json
import torch
import config as cfg
import trt_pose.coco
import trt_pose.models
from torch2trt import TRTModule
import torchvision.transforms as transforms
from trt_pose.parse_objects import ParseObjects
#from trt_pose.draw_objects import DrawObjects


ASSET_DIR = os.environ['HOME'] + '/trt_pose/tasks/human_pose/'
OPTIMIZED_MODEL = ASSET_DIR + 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'

model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

with open(ASSET_DIR + 'human_pose.json', 'r') as f:
    human_pose = json.load(f)

class DrawObjects(object):

    def __init__(self, topology):
        self.topology = topology
        self.body_labels = cfg.body_dict 

    def __call__(self, image, object_counts, obj, normalized_peaks, color):
        topology = self.topology
        height = image.shape[0]
        width = image.shape[1]

        K = topology.shape[0]
        count = int(object_counts[0])
        body_dict = {}
        C = obj.shape[0]
        for j in range(C):
            k = int(obj[j])
            if k >= 0:
                peak = normalized_peaks[0][j][k]
                x = round(float(peak[1]) * width)
                y = round(float(peak[0]) * height)
                cv2.circle(image, (x, y), 3, color, 2)
                body_dict[self.body_labels[j]] = (x,y)

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

        return body_dict

topology = trt_pose.coco.coco_category_to_topology(human_pose)
parse_objects = ParseObjects(topology)
draw_objects = DrawObjects(topology)

def preprocess(image):
    global device
    device = torch.device('cuda')
    image = cv2.resize(image, cfg.input_size)
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
    color = (112,107,222)
    body_list = []
    for obj in objects[0]:
        body_list.append(draw_objects(image, counts, obj, peaks, color))
    return image, body_list

