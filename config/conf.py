images_dir = '/home/funk/Pictures/test_pose'
csv_path = 'data/data.csv'
pose_model = 'models/pose.tflite'
body_dict = {0:'head', 1: 'neck', 2: 'lshoulder', 3:'lelbow',
             4:'lwrist', 5:'rshoulder', 6:'relbow', 7:'rwrist',
             8:'lhip', 9:'lknee', 10:'lankle', 11:'rhip', 12:'rknee', 13:'rankle'}
classifier_model = 'models/classifier.sav'
