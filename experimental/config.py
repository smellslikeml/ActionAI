import os
import json

w = 1280 #1024
h = 720 #768
fps = 25 #30
window = 3
input_size = (224, 224)
secondary = False
log = False
video = True
faces = False
display = False 
annotate = True
learning_rate = 1e-4
max_persons = 2
overlay = False
boxes = False

body_dict = {0:'nose', 1: 'lEye', 2: 'rEye', 3:'lEar', 4:'rEar', 5:'lShoulder', 6:'rShoulder', 
               7:'lElbow', 8:'rElbow', 9:'lWrist', 10:'rWrist', 11:'lHip', 12:'rHip', 13:'lKnee', 14:'rKnee',
              15:'lAnkle', 16:'rAnkle', 17:'neck'}
body_idx = dict([[v,k] for k,v in body_dict.items()])
pose_vec_dim = 2 * len(body_dict)

activity_dict = {'left': '', 
                 'right': '', 
                 'up': '', 
                 'down': '', 
                 'd1L':  '', 
                 'd1R': '', 
                 'select': '', 
                 'start': '', 
                 'cross': 'extension', 
                 'circle': 'curl', 
                 'triangle': 'raise', 
                 'square': 'press', 
                 'jLbutton': '', 
                 'jRbutton': ''}

activity_list = sorted([x for x in activity_dict.values() if x])
idx_dict = {x:idx for idx, x in enumerate(activity_list)}
activity_idx = {idx : activity for idx, activity in enumerate(activity_list)}

ASSET_DIR = os.environ['HOME'] + '/trt_pose/tasks/human_pose/'

with open(ASSET_DIR + 'human_pose.json', 'r') as f:
    human_pose = json.load(f)
