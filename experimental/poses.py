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


ASSET_DIR = os.environ['HOME'] + '/trt_pose/tasks/human_pose/'
OPTIMIZED_MODEL = ASSET_DIR + 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'

model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

with open(ASSET_DIR + 'human_pose.json', 'r') as f:
    human_pose = json.load(f)

class ListHumans(object):
    def __init__(self, body_labels=cfg.body_dict):
        self.body_labels = body_labels

    def __call__(self, objects, normalized_peaks):

        pose_list = []
        for obj in objects[0]:
            pose_dict = {}
            C = obj.shape[0]
            for j in range(C):
                k = int(obj[j])
                if k >= 0:
                    peak = normalized_peaks[0][j][k]
                    x = round(float(peak[1]) * cfg.w)
                    y = round(float(peak[0]) * cfg.h)
                    #cv2.circle(image, (x, y), 3, color, 2)
                    pose_dict[self.body_labels[j]] = (x,y)
            pose_list.append(pose_dict)

        return pose_list

topology = trt_pose.coco.coco_category_to_topology(human_pose)
parse_objects = ParseObjects(topology)
humans = ListHumans()

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
    #color = (112,107,222)  # make dictionary from obj id to cmap
    pose_list = humans(objects, peaks)
    return image, pose_list

