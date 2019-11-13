#!/usr/bin/env python3

import time
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
from keras.models import load_model


interpreter = tf.contrib.lite.Interpreter(model_path="./models/model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def run_inference(image):
    t = time.time()
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    elapsed = time.time() - t
    return interpreter.get_tensor(output_details[0]['index']), elapsed

cam = cv2.VideoCapture(0)
M = np.float32([[1, 0, 0],[0, 1, -15]])

y_dict = {0:'stand', 1:'squat', 2:'deadlift'}

#Load pickled model
loaded_model = load_model('./models/lstm.h5')

N = 5
q = deque(maxlen=N)

def ffwd_n_avg(q): 
    for idx, val in enumerate(q[-1]): 
        if not val: 
            q[-1][idx] = q[-2][idx] 
    return q

count_dict = {'squat': 0, 'deadlift': 0, 'stand': 0}
while True:
    ret_val, image = cam.read()
    image = cv2.resize(image,(192,192),3)
    image = image.reshape((1,192,192,3))
    image = image.astype(np.float32)
    
    output_data, elapsed = run_inference(image)

    feature_vec = np.zeros(28)

    for kp in range(14):
        blf = output_data[:,:,:,kp]
        max_idx = np.argmax(blf)
        coords = divmod(max_idx, 96)
        feature_vec[2*kp:2*kp+2] = coords

    q.append(feature_vec)

    if len(q) == N:
        ff = np.expand_dims(np.concatenate(list(ffwd_n_avg(q)), axis=0).reshape(28, N), axis=0)
        move_pred = loaded_model.predict(ff)
        move_pred = np.argmax(move_pred[0])
        count_dict[y_dict[move_pred]] += 1

    im = np.sum(output_data, axis=3).reshape((96, 96))[:,::-1] * 2
    im = cv2.warpAffine(im, M, (96, 96))
    im = cv2.resize(im, (1016, 1856), interpolation=cv2.INTER_CUBIC)

    cv2.putText(im,
            "S: %i D: %i St: %i" % (count_dict['squat'], count_dict['deadlift'], count_dict['stand']),
                (100, 1500),  cv2.FONT_HERSHEY_SIMPLEX, 3.0,
                (255, 255, 255), 10)
    try:
        cv2.putText(im,
                "LegDay",
                    (100, 100),  cv2.FONT_HERSHEY_SIMPLEX, 3.0,
                    (255, 255, 255), 10)
    except:
        pass

    cv2.namedWindow('window', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('window', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('window', im)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
