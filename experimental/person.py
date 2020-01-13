import cv2
import utils
import config as cfg
import numpy as np
from collections import deque

class PersonTracker(object):
    def __init__(self, expiration=5):
        self.count = 0
        self.activity = ['walk']
        self.expiration = expiration
        self.id = utils.id_gen()
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
        ft_vec = np.zeros(cfg.pose_vec_dim)
        for ky in pose_dict:
            idx = cfg.body_idx[ky]
            ft_vec[2 * idx: 2 * (idx + 1)] = 2 * (np.array(pose_dict[ky]) - np.array(self.centroid)) / np.array((self.h, self.w))
        self.q.append(ft_vec)
        return

    def annotate(self, image, offset=50):
        x1, y1, x2, y2 = self.bbox
        image = cv2.rectangle(image, (x1 - offset, y1 - offset), (x2 + offset, y2 + offset), (195, 195, 89), 2) 
        image = cv2.putText(image, self.activity.upper(), (x1 - offset + 10, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (151, 187, 106), 2) 
        image = cv2.putText(image, self.id, (x1 - offset + 10, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (151, 187, 106), 2) 
        image = cv2.drawMarker(image, self.centroid, (223, 183, 190), 0, 30, 3) 
        return image


