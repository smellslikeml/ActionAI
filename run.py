import numpy as np
import tensorflow as tf
import time
import cv2

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
    print('Time elapsed: ', elapsed, feature_vec)


