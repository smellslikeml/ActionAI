import os
import cv2

# From Github: https://github.com/opencv/opencv/tree/master/data/haarcascades
ROOT_DIR = os.environ['HOME'] + '/ActionAI/experimental'
MDL_DIR = ROOT_DIR + '/models/'
mdl_path = os.path.join(MDL_DIR, 'haarcascade_frontalface_alt.xml')

class FaceDetector():
    def __init__(self, nbs=5, scale=1.1, inset=150, min_size=10, model_file=mdl_path):
        self.detector = cv2.CascadeClassifier(mdl_path)
        self.nbs = nbs
        self.scale = scale
        self.inset = inset
        self.min_size = (min_size, min_size)

    def run_inference(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.objs = self.detector.detectMultiScale(gray, scaleFactor=self.scale, minNeighbors=self.nbs, minSize=self.min_size)
        return

    def process_frame(self, img):
        self.run_inference(img)
        img_copy = img.copy()
        for idx, (x, y, w, h) in enumerate(self.objs):
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)
            img_copy[idx * self.inset:(idx + 1) * self.inset, :self.inset, :] = cv2.resize(img[y:y+h, x:x+w, :], (self.inset, self.inset))
        return img_copy

if __name__ == "__main__":
    detector = FaceDetector()
    cap = cv2.VideoCapture(1)
    while True:
        ret, image = cap.read()
        face = detector.process_frame(image)

        cv2.imshow("Objects found", face)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break


    cap.release()
    cv2.destroyAllWindows()
