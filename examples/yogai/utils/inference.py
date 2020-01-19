#!/usr/bin/env python3
import os
import time
import cv2
import pickle
import numpy as np
from glob import glob
import tensorflow as tf
from collections import deque
from keras.models import load_model

class motionClassifier(object):
    def __init__(self, model_path='./models/model.tflite', pose_model='tree'):
        self.model_path = model_path
        self.interpreter = tf.contrib.lite.Interpreter(model_path=self.model_path)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        _, self.input_dim, _, _ = self.input_details[0]['shape']
        _, self.mp_dim, _, self.ky_pt_num = self.output_details[0]['shape']

        self.motion_path = glob('./models/motion_*')
        self.N = self.motion_path.rstrip('.h5').split('_')[-1]
        self.motion_model = load_model(self.motion_path[-1])

        self.q = deque(maxlen=self.N)

        self.pose_model = pose_model
        if self.pose_model == 'tree':
            self.pose_model = pickle.load(open('./models/yoga_poses.sav', 'rb'))
        else:
            self.pose_model = load_model('./models/yoga_poses_keras.h5')

        self.move_lst = os.listdir('./data/legday/')
        self.move_dict = {idx:val for idx,val in enumerate(self.move_lst)}

        self.pose_lst = os.listdir('./data/yoga/')
        self.pose_dict = {idx:val for idx,val in enumerate(self.pose_lst)}

        self.count_dict = {mv:0 for mv in self.move_lst}

    def ft_ext(self, image):
        t = time.time()
        self.interpreter.set_tensor(self.input_details[0]['index'], image)
        self.interpreter.invoke()
        self.elapsed = time.time() - t
        self.result = self.interpreter.get_tensor(self.output_details[0]['index'])
        res = self.result.reshape(1, self.mp_dim**2, self.ky_pt_num)
        max_idxs = np.argmax(res, axis=1)
        coords = list(map(lambda x: divmod(x, self.mp_dim), max_idxs))
        self.feature_vec = np.concatenate(coords).reshape(2 * self.ky_pt_num, 1)
        self.q.append(self.feature_vec)

    def ffwd_n_avg(self):
        for idx in np.where(self.q[-1] == 0)[0]:
            self.q[-1][idx] = self.q[-2][idx]
        return self.q

    def motion(self):
        ff = np.expand_dims(np.concatenate(list(self.ffwd_n_avg()), axis=0).reshape(2 * self.ky_pt_num, self.N), axis=0)
        move_pred = self.motion_model.predict(ff)
        return np.argmax(move_pred[0])

    def pose(self):
        if self.pose_model == 'tree':
            return str(self.pose_model.predict(self.feature_vec))
        else:
            pose_pred = self.pose_model.predict(self.feature_vec)
            return self.pose_dict[np.argmax(pose_pred[0])]




if __name__ == '__main__':
    from screen_size import getScreenDims
    sc_dims = tuple(getScreenDims())

    legday = True

    mC = motionClassifier()
    cam = cv2.VideoCapture(0)
    M = np.float32([[1, 0, 0],[0, 1, -12]])
    while True:
        ret_val, image = cam.read()
        image = cv2.resize(image, (mC.input_dim, mC.input_dim), 3)
        image = np.expand_dims(image, axis=0).astype(np.float32)
        mC.ft_ext(image)

        if len(mC.q) == mC.N:
            move_pred = mC.motion()
            mC.count_dict[mC.move_dict[move_pred]] += 1

            im = np.sum(mC.result, axis=3).reshape((mC.mp_dim, mC.mp_dim))[:,::-1] * 1.2
            im = cv2.warpAffine(im, M, (mC.mp_dim, mC.mp_dim))
            im = cv2.resize(im, sc_dims, interpolation=cv2.INTER_CUBIC)

            if legday:
                cv2.putText(im,
                        "S: %i D: %i St: %i tm: %f" % (mC.count_dict['squat'], mC.count_dict['deadlift'], mC.count_dict['stand'], mC.elapsed),
                            (100, 1500),  cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                            (255, 255, 255), 10)
                cv2.putText(im,
                        "LegDay",
                            (100, 100),  cv2.FONT_HERSHEY_SIMPLEX, 3.0,
                            (255, 255, 255), 10)
            else:
                pose_pred = mC.pose()
                cv2.putText(im, "YogAI", 
                            (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3.0,
                            (255, 255, 255), 15)
                cv2.putText(im, "%s" % pose_pred,
                            (100, 1500), cv2.FONT_HERSHEY_SIMPLEX, 3.0,
                            (255, 255, 255), 15)

            cv2.namedWindow('window', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('window', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow('window', im)
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
