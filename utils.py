import string
import random
from operator import itemgetter

body_labels = {
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
body_idx = dict([[v, k] for k, v in body_labels.items()])


def id_gen(size=6, chars=string.ascii_uppercase + string.digits):
    """
    https://pythontips.com/2013/07/28/generating-a-random-string/
    input: id_gen(3, "6793YUIO")
    output: 'Y3U'
    """
    return "".join(random.choice(chars) for x in range(size))


def IOU(boxA, boxB):
    # pyimagesearch: determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def get_bbox(kp_list):
    bbox = []
    for aggs in [min, max]:
        for idx in range(2):
            bound = aggs(kp_list, key=itemgetter(idx))[idx]
            bbox.append(bound)
    return bbox


def id_gen(size=6, chars=string.ascii_uppercase + string.digits):
    """
    https://pythontips.com/2013/07/28/generating-a-random-string/
    input: id_gen(3, "6793YUIO")
    output: 'Y3U'
    """
    return "".join(random.choice(chars) for x in range(size))
