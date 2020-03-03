import sys
import cv2
import csv
import time
import numpy as np
from operator import itemgetter


#import poses
from mtcnn.mtcnn import MTCNN
import utils
import person
import model as mdl
import config as cfg

timestamp = int(time.time() * 1000)

detector = MTCNN()

if cfg.secondary:
    from faces import FaceDetector
    detector = FaceDetector()

if cfg.log:
    dat = os.path.base_name(sys.argv[1])
    dataFile = open('data/logs/{}'.format(dat.replace('.mp4', '.csv')), 'w')
    newFileWriter = csv.writer(dataFile)

trackers = []
cap = utils.source_capture(sys.argv[1])
img = utils.img_obj()

while True:

    ret, frame = cap.read()
    bboxes = []
    if ret:

        #image, pose_list = poses.inference(frame)
        faces = detector.detect_faces(frame)
        
        for face in faces:
            if face:
                #bbox = utils.get_bbox(list(body.values()))
                bbox = face['box']
                bboxes.append(bbox)

        trackers = utils.update_trackers(trackers, bboxes)

        #if cfg.log:
        #    newFileWriter.writerow([tracker.activity] + list(np.hstack(list(tracker.q)[:cfg.window])))

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

            ann_trackers = sorted([(tracker, tracker.centroid[0]) for tracker in ann_trackers], key=itemgetter(1))
            ann_trackers = [tup[0] for tup in ann_trackers]
            print(ann_trackers)
            #for idx, tracker in enumerate(ann_trackers):
            #    tracker.skeleton_dict = cfg.cmap_list[idx]
            #    image = img.annotate(tracker, image, boxes=cfg.boxes)

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
