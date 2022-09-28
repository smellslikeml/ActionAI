import config as cfg


class ListHumans(object):
    def __init__(self, body_labels=cfg.body_dict):
        self.body_labels = body_labels

    def __call__(self, objects, normalized_peaks):

        pose_list = []
        for obj in objects[0]:
            pose_dict = {}
            C = obj.shape[0]
            for j in range(C):
                k = int(obj[j])
                if k >= 0:
                    peak = normalized_peaks[0][j][k]
                    x = round(float(peak[1]) * w)
                    y = round(float(peak[0]) * h)
                    # cv2.circle(image, (x, y), 3, color, 2)
                    pose_dict[self.body_labels[j]] = (x, y)
            pose_list.append(pose_dict)

        return pose_list
