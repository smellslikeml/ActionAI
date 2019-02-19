#!/usr/bin/env python3
import time
import argparse
import logging
import numpy as np
import tensorflow as tf
import time
import cv2

interpreter = tf.contrib.lite.Interpreter(model_path="./models/model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Logging stuff
logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def run_inference(image):
    t = time.time()
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    elapsed = time.time() - t
    return interpreter.get_tensor(output_details[0]['index']), elapsed

#Setting up webcam
cam = cv2.VideoCapture(0)
M = np.float32([[1, 0, 0],[0, 1, -15]])


if __name__ == '__main__':
    #TODO: implement the --bootstrap and --flow options
    parser = argparse.ArgumentParser(description='YogAI run basic inference without webcam')
    parser.add_argument('--pose_model', type=str, default='./models/yoga_poses.sav', help='a previously trained pose classifier')
    parser.add_argument('--bootstrap', type=bool, default=False, help='set as True to label new pose samples (default=False) ')
    parser.add_argument('--heatmap', type=bool, default=False, help='visualize the generated pose estimator heatmap and pose classification')
    parser.add_argument('--flow', type=str, default='demo', help='pose routine: choose from demo, random, or create your own as a list of poses ie. ["pose1", "pose2", "pose3"]')
    args = parser.parse_args()
    

    #Load pickled model
    loaded_model = pickle.load(open(args.pose_model, 'rb'))

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
        pose_pred = loaded_model.predict(feature_vec.reshape(1, -1))

        if args.heatmap == True:
	    im = np.sum(output_data, axis=3).reshape((96, 96))[:,::-1] * 2 
	    im = cv2.warpAffine(im, M, (96, 96))
	    im = cv2.resize(im, (1016, 1856), interpolation=cv2.INTER_CUBIC)
	    cv2.putText(im, "%s" % str(pose_pred[0]),
			(100, 1500),  cv2.FONT_HERSHEY_SIMPLEX, 3.0,
			(220, 223, 154), 15) 

	    cv2.putText(im,
		    "YogAI", 
			(100, 100),  cv2.FONT_HERSHEY_SIMPLEX, 3.0,
			(255, 255, 255), 15) 

	    cv2.namedWindow('window', cv2.WND_PROP_FULLSCREEN)
	    cv2.setWindowProperty('window', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
	    cv2.imshow('window', im) 
	    if cv2.waitKey(1) == 27: 
		break
        else:
            print('Time elapsed: ', elapsed, feature_vec)
    cv2.destroyAllWindows()

