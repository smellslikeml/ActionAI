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

#Loading model
interpreter = tf.contrib.lite.Interpreter(model_path="./models/model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


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
    parser.add_argument('--folder', type=str, default='./poses/')
    args = parser.parse_args()

    dirs = os.listdir(args.folder)
    with open("./poses/poses.csv", "w") as fl:
        writer = csv.writer(fl)
        for ydir in dirs:
            pose_dir = args.folder + ydir + '/'
            files_grabbed = glob.glob(os.path.join(pose_dir, '*'))
            for i, fl in enumerate(files_grabbed):
                output_data, elapsed = run_inference(fl)
                logger.info('inference image #%d: %s in %.4f seconds.' % (i, fl, elapsed))

                feature_vec = np.zeros(28)
                for kp in range(14):
                    blf = output_data[:,:,:,kp]
                    max_idx = np.argmax(blf)
                    coords = divmod(max_idx, 96)
                    feature_vec[2*kp:2*kp+2] = coords
                row = feature_vec.tolist()
                row.append(ydir)
                writer.writerow(row)
