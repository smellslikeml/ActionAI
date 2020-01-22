import cv2
import utils
import config as cfg
import numpy as np
from collections import deque

class PersonTracker(object):
    def __init__(self, expiration=5):
        self.count = 0
        self.activity = ''
        self.expiration = expiration
        self.id = utils.id_gen()
        self.q = deque(maxlen=10)
        return
        
    def set_bbox(self, bbox):
        self.bbox = bbox
        x1, y1, x2, y2 = bbox
        self.h = 1e-6 + x2 - x1
        self.w = 1e-6 + y2 - y1
        self.centroid = tuple(map(int, (x1 + self.h / 2, \
                                        y1 + self.w / 2)))
        return

    def update_pose(self, pose_dict):
        ft_vec = np.zeros(cfg.pose_vec_dim)
        for ky in pose_dict:
            idx = cfg.body_idx[ky]
            ft_vec[2 * idx: 2 * (idx + 1)] = 2 * (np.array(pose_dict[ky]) -  \
                                                  np.array(self.centroid)) / \
                                                  np.array((self.h, self.w))
        self.q.append(ft_vec)
        return


