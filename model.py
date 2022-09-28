import os
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
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical

w = 640
h = 480
fps = 25
window = 3
input_size = (224, 224)
secondary = True
log = False
video = True
faces = False
display = False
learning_rate = 1e-4
max_persons = 2
overlay = False
boxes = False


activity_dict = {
    "left": "",
    "right": "",
    "up": "",
    "down": "",
    "d1L": "",
    "d1R": "",
    "select": "",
    "start": "",
    "cross": "extension",
    "circle": "curl",
    "triangle": "raise",
    "square": "press",
    "jLbutton": "",
    "jRbutton": "",
}

activity_list = sorted([x for x in activity_dict.values() if x])
idx_dict = {x: idx for idx, x in enumerate(activity_list)}
activity_idx = {idx: activity for idx, activity in enumerate(activity_list)}

ASSET_DIR = os.environ["HOME"] + "/trt_pose/tasks/human_pose/"

with open(ASSET_DIR + "human_pose.json", "r") as f:
    human_pose = json.load(f)

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
pose_vec_dim = 2 * len(body_labels)

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


def lstm_model():
    model = Sequential()
    model.add(
        LSTM(
            16,
            dropout=0.2,
            recurrent_dropout=0.2,
            input_shape=(pose_vec_dim, window),
        )
    )
    model.add(Dense(16, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(len(activity_dict), activation="softmax"))
    print(model.summary())
    return model
