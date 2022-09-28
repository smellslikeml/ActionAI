import os
import sys
import cv2
import csv
import time
import numpy as np

from model import *
from utils import *

import os
import json

w = 640
h = 480
fps = 25
window = 3
input_size = (224, 224)
secondary = True
log = False
video = True
faces = False
display = False
annotate = True
learning_rate = 1e-4
max_persons = 2
overlay = False
boxes = False

if __name__ == "__main__":
    source = sys.argv[1]
    source = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(source)

    w = int(cap.get(3))
    h = int(cap.get(4))

    fourcc_cap = cv2.VideoWriter_fourcc(*"MJPG")
    cap.set(cv2.CAP_PROP_FOURCC, fourcc_cap)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    if log:
        activity = os.path.basename(source)
        dataFile = open("data/{}.csv".format(activity), "w")
        newFileWriter = csv.writer(dataFile)

    if video:
        # Define the codec and create VideoWriter object
        name = "out.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(name, fourcc, 30.0, (w, h))

    if secondary:
        import tensorflow as tf

        secondary_model = tf.keras.models.load_model("models/lstm_spin_squat.h5")
        window = 3
        pose_vec_dim = 36
        motion_dict = {0: "spin", 1: "squat"}

    trackers = []
    while True:

        ret, frame = cap.read()
        bboxes = []
        if ret:

            image, pose_list = inference(frame)
            for body in pose_list:
                bbox = get_bbox(list(body.values()))
                bboxes.append((bbox, body))

            track_boxes = [tracker.bbox for tracker in trackers]
            matched, unmatched_trackers, unmatched_detections = tracker_match(
                track_boxes, [b[0] for b in bboxes]
            )

            for idx, jdx in matched:
                trackers[idx].set_bbox(bboxes[jdx][0])
                trackers[idx].update_pose(bboxes[jdx][1])

            for idx in unmatched_detections:
                try:
                    trackers.pop(idx)
                except:
                    pass

            for idx in unmatched_trackers:
                person = PersonTracker()
                person.set_bbox(bboxes[idx][0])
                person.update_pose(bboxes[idx][1])
                trackers.append(person)

            if secondary:
                for tracker in trackers:
                    if len(tracker.q) >= 3:
                        sample = np.array(list(tracker.q)[:3])
                        sample = sample.reshape(1, pose_vec_dim, window)
                        pred_activity = motion_dict[
                            np.argmax(secondary_model.predict(sample)[0])
                        ]
                        tracker.activity = pred_activity
                        image = tracker.annotate(image)
                        print(pred_activity)

            if log:
                for tracker in trackers:
                    if len(tracker.q) >= 3:
                        newFileWriter.writerow(
                            [activity] + list(np.hstack(list(tracker.q)[:3]))
                        )

            if video:
                out.write(image)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()

    try:
        dataFile.close()
    except:
        pass

    try:
        out.release()
    except:
        pass
