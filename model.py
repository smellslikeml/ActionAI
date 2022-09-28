import cv2
import json
import torch
import PIL.Image

import trt_pose.coco
import trt_pose.models
from torch2trt import TRTModule
import torchvision.transforms as transforms
from trt_pose.parse_objects import ParseObjects

import annotate

model_w = 224
model_h = 224

ASSET_DIR = "models/"
OPTIMIZED_MODEL = ASSET_DIR + "resnet18_baseline_att_224x224_A_epoch_249_trt.pth"

body_labels = {
    0: "nose",
    1: "lEye",
    2: "rEye",
    3: "lEar",
    4: "rEar",
    5: "lShoulder",
    6: "rShoulder",
    7: "lElbow",
    8: "rElbow",
    9: "lWrist",
    10: "rWrist",
    11: "lHip",
    12: "rHip",
    13: "lKnee",
    14: "rKnee",
    15: "lAnkle",
    16: "rAnkle",
    17: "neck",
}
body_idx = dict([[v, k] for k, v in body_labels.items()])

with open(ASSET_DIR + "human_pose.json", "r") as f:
    human_pose = json.load(f)

model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device("cuda")

topology = trt_pose.coco.coco_category_to_topology(human_pose)
parse_objects = ParseObjects(topology)
draw_objects = annotate.DrawObjects(topology, body_labels)


def preprocess(image):
    global device
    device = torch.device("cuda")
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
    counts, objects, peaks = parse_objects(
        cmap, paf
    )  # , cmap_threshold=0.15, link_threshold=0.15)
    body_dict = draw_objects(image, counts, objects, peaks)
    return image, body_dict
