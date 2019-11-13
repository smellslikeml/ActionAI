#!/usr/bin/env python3
import argparse
import logging
import time
import glob
import os

import numpy as np
import tensorflow as tf
import cv2
import csv

# Loading model
print('Loading model...')
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


def load_image(path):
    image = cv2.imread(path)
    image = cv2.resize(image,(192,192),3)
    image = image.reshape((1,192,192,3))
    image = image.astype(np.float32)
    return image


def run_inference(image):
    t = time.time()
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    elapsed = time.time() - t
    return interpreter.get_tensor(output_details[0]['index']), elapsed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YogAI label pose samples for classifier')
    parser.add_argument('--folder', type=str, default='../data/yoga/')
    args = parser.parse_args()

    dirs = os.listdir(args.folder)
    with open(args.folder + "poses.csv" , "w") as fl:
        writer = csv.writer(fl)
        for ydir in dirs:
            pose_dir = args.folder + ydir + '/'
            files_grabbed = glob.glob(os.path.join(pose_dir, '*[!.txt]'))
            for i, file in enumerate(files_grabbed):
                try:
                    img = load_image(file)
                    output_data, elapsed = run_inference(img)
                    logger.info('inference image #%d: %s in %.4f seconds.' % (i, file, elapsed))

                    feature_vec = np.zeros(28)
                    for kp in range(14):
                        blf = output_data[:,:,:,kp]
                        max_idx = np.argmax(blf)
                        coords = divmod(max_idx, 96)
                        feature_vec[2*kp:2*kp+2] = coords
                    row = feature_vec.tolist()
                    row.append(ydir)
                    writer.writerow(row)
                except (ValueError, cv2.error) as e:
                    print(file)
