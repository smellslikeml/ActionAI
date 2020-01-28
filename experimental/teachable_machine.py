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
import config as cfg
import control as ps3

timestamp = int(time.time() * 1000)

if cfg.secondary:
    secondary_model = mdl.lstm_model()
    secondary_model.compile(loss='categorical_crossentropy', 
                            optimizer=mdl.RMSprop(lr=cfg.learning_rate), 
                            metrics=['accuracy'])

if cfg.faces:
    from faces import FaceDetector
    detector = FaceDetector()

if cfg.log:
    dataFile = open('data/logs/{}.csv'.format(timestamp), 'w')
    newFileWriter = csv.writer(dataFile)

if cfg.video:
    # Define the codec and create VideoWriter object
    name = 'data/videos/{}.mp4'.format(timestamp)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(name, fourcc, cfg.fps, (cfg.w, cfg.h))

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
        #print([(tracker.id, np.vstack(tracker.q)) for tracker in trackers])

        for tracker in trackers:
            activity = [cfg.activity_dict[x] for x in ps3.getButton()]
            if len(tracker.q) >= cfg.window and cfg.secondary:
                sample = np.array(list(tracker.q)[:cfg.window])
                sample = sample.reshape(1, cfg.pose_vec_dim, cfg.window)
                if activity:
                    for act in activity:
                        a_vec = np.expand_dims(mdl.to_categorical(cfg.idx_dict[act], len(cfg.activity_dict)), axis=0)
                        secondary_model.fit(sample, a_vec, batch_size=1, epochs=1, verbose=1)
                    tracker.activity = activity
                else:
                    try:
                        pred_activity = cfg.activity_idx[np.argmax(secondary_model.predict(sample)[0])]
                        tracker.activity = list(pred_activity)
                    except KeyError:
                        print('error')

                print(tracker.activity)

            if cfg.log:
                newFileWriter.writerow([tracker.activity] + list(np.hstack(list(tracker.q)[:cfg.window])))

        if cfg.annotate:
            if cfg.faces:
                # TODO: decouple from annotation, 
                # add face feature to person class
                image = detector.process_frame(image)

            ann_trackers = []
            for tracker in trackers:
                if len(tracker.q) >= cfg.window:
                    ann_trackers.append(tracker)

            ann_trackers = [(tracker, np.prod((tracker.w, tracker.h))) for tracker in ann_trackers]
            ann_trackers = [tup[0] for tup in sorted(ann_trackers, key=itemgetter(1), reverse=True)][:cfg.max_persons]
            if not cfg.overlay:
                image = np.zeros_like(image).astype('uint8')
            for tracker in ann_trackers:
                image = img.annotate(tracker, image, boxes=cfg.boxes)

        if cfg.video:
            out.write(image)

        if cfg.display:
            cv2.imshow(sys.argv[1], image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
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
