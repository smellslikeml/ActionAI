import cv2
import utils
import config as cfg
import numpy as np
from collections import deque

class PersonTracker(object):
    def __init__(self, expiration=5):
        self.count = 0
        self.eps = 1e-6
        self.activity = ''
        self.expiration = expiration
        self.id = utils.id_gen()
        self.q = deque(maxlen=10)
        self.cubit_q = deque(maxlen=50)
        return
        
    def set_bbox(self, bbox):
        '''
        Used to calculate
        bounding box for
        tracking and annotating
        person instances
        '''
        self.bbox = bbox
        x1, y1, x2, y2 = bbox
        self.h = self.eps + x2 - x1
        self.w = self.eps + y2 - y1
        self.centroid = tuple(map(int, (x1 + self.h / 2, \
                                        y1 + self.w / 2)))
        return

    def set_pose(self, pose_dict):
        '''
        Used to encode pose estimates
        over a time window
        '''
        self.pose_dict = pose_dict
        ft_vec = np.zeros(cfg.pose_vec_dim)
        for ky in pose_dict:
            idx = cfg.body_idx[ky]
            ft_vec[2 * idx: 2 * (idx + 1)] = 2 * (np.array(pose_dict[ky]) -  \
                                                  np.array(self.centroid)) / \
                                                  np.array((self.h, self.w))
        self.q.append(ft_vec)
        return

    def set_cubit(self, pose_dict):
        '''
        Used to estimate the cubit
        for the purposes of calculating
        distance, in a queue for averaging
        '''
        for side in ['r', 'l']:
            try:
                p1, p2 = list(map(np.array, [pose_dict['{}Elbow'.format(side)], pose_dict['{}Wrist'.format(side)]]))
                self.cubit_q.append(np.linalg.norm(p1 - p2))
            except:
                pass


