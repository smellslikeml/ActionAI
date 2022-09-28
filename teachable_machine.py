import sys
import cv2
import csv
import time
import numpy as np
from operator import itemgetter

import poses
import utils
import person
import model as mdl
import control as ps3

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

body_dict = {
    0: "nose",
    1: "lEye",
    2: "rEye",
    3: "lEar",
    4: "rEar",
    5: "lShoulder",
    6: "rShoulder",
    7: "lElbow",
    8: "rElbow",
    9: "lWrist",
    10: "rWrist",
    11: "lHip",
    12: "rHip",
    13: "lKnee",
    14: "rKnee",
    15: "lAnkle",
    16: "rAnkle",
    17: "neck",
}
body_idx = dict([[v, k] for k, v in body_dict.items()])
pose_vec_dim = 2 * len(body_dict)

activity_dict = {
    "left": "",
    "right": "",
    "up": "",
    "down": "",
    "d1L": "",
    "d1R": "",
    "select": "",
    "start": "",
    "cross": "extension",
    "circle": "curl",
    "triangle": "raise",
    "square": "press",
    "jLbutton": "",
    "jRbutton": "",
}

activity_list = sorted([x for x in activity_dict.values() if x])
idx_dict = {x: idx for idx, x in enumerate(activity_list)}
activity_idx = {idx: activity for idx, activity in enumerate(activity_list)}

ASSET_DIR = os.environ["HOME"] + "/trt_pose/tasks/human_pose/"

with open(ASSET_DIR + "human_pose.json", "r") as f:
    human_pose = json.load(f)

timestamp = int(time.time() * 1000)

if secondary:
    secondary_model = mdl.lstm_model()
    secondary_model.compile(
        loss="categorical_crossentropy",
        optimizer=mdl.RMSprop(lr=learning_rate),
        metrics=["accuracy"],
    )

if faces:
    from faces import FaceDetector

    detector = FaceDetector()

if log:
    dataFile = open("data/logs/{}.csv".format(timestamp), "w")
    newFileWriter = csv.writer(dataFile)

if video:
    # Define the codec and create VideoWriter object
    name = "data/videos/{}.mp4".format(timestamp)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(name, fourcc, fps, (w, h))

trackers = []
cap = utils.source_capture(sys.argv[1])
img = utils.img_obj()


while True:

    ret, frame = cap.read()
    bboxes = []
    if ret:

        image, pose_list = poses.inference(frame)
        for body in pose_list:
            if body:
                bbox = utils.get_bbox(list(body.values()))
                bboxes.append((bbox, body))

        trackers = utils.update_trackers(trackers, bboxes)
        # print([(tracker.id, np.vstack(tracker.q)) for tracker in trackers])

        for tracker in trackers:
            activity = [activity_dict[x] for x in ps3.getButton()]
            if len(tracker.q) >= window and secondary:
                sample = np.array(list(tracker.q)[: window])
                sample = sample.reshape(1, pose_vec_dim, window)
                if activity:
                    for act in activity:
                        a_vec = np.expand_dims(
                            mdl.to_categorical(
                                idx_dict[act], len(activity_dict)
                            ),
                            axis=0,
                        )
                        secondary_model.fit(
                            sample, a_vec, batch_size=1, epochs=1, verbose=1
                        )
                    tracker.activity = activity
                else:
                    try:
                        pred_activity = activity_idx[
                            np.argmax(secondary_model.predict(sample)[0])
                        ]
                        tracker.activity = pred_activity #list(pred_activity)
                        print(pred_activity)
                    except KeyError:
                        print("error")

                print(tracker.activity)

            if log:
                newFileWriter.writerow(
                    [tracker.activity] + list(np.hstack(list(tracker.q)[: window]))
                )

        if annotate:
            if faces:
                # TODO: decouple from annotation,
                # add face feature to person class
                image = detector.process_frame(image)

            ann_trackers = []
            for tracker in trackers:
                if len(tracker.q) >= window:
                    ann_trackers.append(tracker)

            ann_trackers = [
                (tracker, np.prod((tracker.w, tracker.h))) for tracker in ann_trackers
            ]
            ann_trackers = [
                tup[0] for tup in sorted(ann_trackers, key=itemgetter(1), reverse=True)
            ][: max_persons]
            if not overlay:
                image = np.zeros_like(image).astype("uint8")
            #for tracker in ann_trackers:
                #image = img.annotate(tracker, image, boxes=boxes)

        if video:
            out.write(image)

        if display:
            cv2.imshow(sys.argv[1], image)

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
