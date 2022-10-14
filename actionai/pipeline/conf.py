# Choose your classifier
from sklearn.linear_model import LogisticRegression as classifier

# Set source for opencv VideoCapture: usb index, mp4, rtsp
stream = 0

# Set path to staged data, raw images, pickled sklearn model
csv_path = 'data/data.csv'
images_dir = 'path/to/images/dir'
classifier_model = 'models/classifier.sav'

# Set path to pose estimation model asset and body keypoint map
pose_model = 'models/pose.tflite'
body_dict = {0:'head', 1: 'neck', 2: 'lshoulder', 3:'lelbow',
             4:'lwrist', 5:'rshoulder', 6:'relbow', 7:'rwrist',
             8:'lhip', 9:'lknee', 10:'lankle', 11:'rhip', 12:'rknee', 13:'rankle'}
