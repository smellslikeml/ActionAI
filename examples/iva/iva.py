import os
import sys
import cv2
import csv
import time
import numpy as np

from model import *
from utils import *
import config as cfg
from person import PersonTracker

import os
import json

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

    if cfg.log:
        activity = os.path.basename(source)
        dataFile = open("data/{}.csv".format(activity), "w")
        newFileWriter = csv.writer(dataFile)

    if cfg.video:
        # Define the codec and create VideoWriter object
        name = "out.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(name, fourcc, 30.0, (cfg.w, cfg.h))

    if cfg.secondary:
        import tensorflow as tf

        secondary_model = tf.keras.models.load_model("models/lstm_spin_squat.h5")
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
                trackers[idx].set_pose(bboxes[jdx][1])

            for idx in unmatched_detections:
                try:
                    trackers.pop(idx)
                except:
                    pass

            for idx in unmatched_trackers:
                person = PersonTracker()
                person.set_bbox(bboxes[idx][0])
                person.set_pose(bboxes[idx][1])
                trackers.append(person)

            if cfg.secondary:
                for tracker in trackers:
                    if len(tracker.q) >= 3:
                        sample = np.array(list(tracker.q)[:3])
                        sample = sample.reshape(1, cfg.pose_vec_dim, cfg.window)
                        pred_activity = motion_dict[
                            np.argmax(secondary_model.predict(sample)[0])
                        ]
                        tracker.activity = pred_activity
                        #image = tracker.annotate(image)
                        print(pred_activity)

            if cfg.log:
                for tracker in trackers:
                    if len(tracker.q) >= 3:
                        newFileWriter.writerow(
                            [activity] + list(np.hstack(list(tracker.q)[:3]))
                        )

            if cfg.video:
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
