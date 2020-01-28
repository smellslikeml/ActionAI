import os
import cv2
import utils
import random
import config as cfg
import numpy as np
from collections import deque

# From Github: https://github.com/opencv/opencv/tree/master/data/haarcascades
ROOT_DIR = os.environ['HOME'] + '/ActionAI/experimental'
MDL_DIR = ROOT_DIR + '/models/'
face_mdl = os.path.join(MDL_DIR, 'haarcascade_frontalface_alt.xml')


class PersonTracker(object):
    def __init__(self, expiration=5, nbs=5, scale=1.1, inset=150, min_size=10, model_file=face_mdl):
        self.count = 0
        self.eps = 1e-6
        self.activity = []
        self.expiration = expiration
        self.id = utils.id_gen()
        self.q = deque(maxlen=10)
        self.cubit_q = deque(maxlen=50)
        self.skeleton_color = tuple([random.randint(0, 255) for _ in range(3)]) #skeleton_color

        self.faces = []
        self.detector = cv2.CascadeClassifier(model_file)
        self.nbs = nbs
        self.scale = scale
        self.inset = inset
        self.min_size = (min_size, min_size)

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

    def get_face(self, image):
        x1, y1, x2, y2 = self.bbox
        body_crop = image[x1:x2, y1:y2, :]
        gray = cv2.cvtColor(body_crop, cv2.COLOR_BGR2GRAY)
        self.objs = self.detector.detectMultiScale(gray, scaleFactor=self.scale, minNeighbors=self.nbs, minSize=self.min_size)
        for idx, (x, y, w, h) in enumerate(self.objs):
            self.faces.append(body_crop[x:x+w, y:y+h, :])
