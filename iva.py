import os
import sys
import cv2
import csv
import time
import numpy as np
from collections import deque
from scipy.optimize import linear_sum_assignment

from model import *
from utils import *
import experimental.config as cfg


def tracker_distance(prev_roi, curr_roi, metric=IOU):
    distance_matrix = np.zeros((len(prev_roi), len(curr_roi)), dtype=np.float32)
    for t, trk in enumerate(prev_roi):
        for d, det in enumerate(curr_roi):
            distance_matrix[t, d] = metric(trk, det)
    return distance_matrix


def tracker_match(prev_roi, curr_roi, threshold=0.3):
    """
    From previously tracked regions of interest and current detections,
    output matched detections, unmatched trackers, unmatched detections.
    """

    distance_matrix = tracker_distance(prev_roi, curr_roi)

    # Hungarian algorithm (also known as Munkres algorithm)
    rowIdxs, colIdxs = linear_sum_assignment(-distance_matrix)

    unmatched_trackers = list(set(range(len(prev_roi))).difference(rowIdxs))
    unmatched_detections = list(set(range(len(curr_roi))).difference(colIdxs))

    matches = []
    for m in list(zip(rowIdxs, colIdxs)):
        if distance_matrix[m[0], m[1]] < threshold:
            unmatched_trackers.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(np.array(m).reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class PersonTracker(object):
    def __init__(self):
        self.id = id_gen()
        self.q = deque(maxlen=10)
        return

    def set_bbox(self, bbox):
        self.bbox = bbox
        x1, y1, x2, y2 = bbox
        self.h = 1e-6 + x2 - x1
        self.w = 1e-6 + y2 - y1
        self.centroid = tuple(map(int, (x1 + self.h / 2, y1 + self.w / 2)))
        return

    def update_pose(self, pose_dict):
        ft_vec = np.zeros(2 * len(body_labels))
        for ky in pose_dict:
            idx = body_idx[ky]
            ft_vec[2 * idx : 2 * (idx + 1)] = (
                2
                * (np.array(pose_dict[ky]) - np.array(self.centroid))
                / np.array((self.h, self.w))
            )
        self.q.append(ft_vec)
        return

    def annotate(self, image):
        x1, y1, x2, y2 = self.bbox
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
        image = cv2.putText(
            image,
            self.activity,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
        )
        image = cv2.drawMarker(image, self.centroid, (255, 0, 0), 0, 30, 4)
        return image


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
        out = cv2.VideoWriter(name, fourcc, 30.0, (w, h))

    if cfg.secondary:
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

            if cfg.secondary:
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
