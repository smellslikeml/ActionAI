# taken from: https://github.com/NVIDIA-AI-IOT/trt_pose/blob/master/tasks/human_pose/live_demo.ipynb

import json
import torch

import torch2trt
import trt_pose.coco
import trt_pose.models
from torch2trt import TRTModule

from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects

import cv2
import torchvision.transforms as transforms
import PIL.Image

# Set variables 
WIDTH = 224
HEIGHT = 224
MODEL_WEIGHTS = 'models/resnet18_baseline_att_224x224_A_epoch_249.pth'
OPTIMIZED_MODEL = 'models/resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
with open('models/human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)
parse_objects = ParseObjects(topology)
draw_objects = DrawObjects(topology)

num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])

# Load original model
model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
model.load_state_dict(torch.load(MODEL_WEIGHTS))

# Use torch2trt to optimize and save
data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()
model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)
torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)

# Load optimized model
model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))

# Optional: uncomment to benchmark fps
"""
import time

t0 = time.time()
torch.cuda.current_stream().synchronize()
for i in range(50):
    y = model_trt(data)
torch.cuda.current_stream().synchronize()
t1 = time.time()

print(50.0 / (t1 - t0))
"""

# Preprocessing and inference methods
mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

def preprocess(image):
    global device
    device = torch.device('cuda')
    image = cv2.resize(image, (HEIGHT, WIDTH))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

def predict(image):
    data = preprocess(image)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
    print("Found counts: ", counts)
    draw_objects(image, counts, objects, peaks)

if __name__ == "__main__":
    test_image_path = "test.jpg"
    test_image = cv2.imread(test_image_path)
    predict(test_image)
    cv2.imwrite("out_test.jpg", test_image)
    
