import os
import cv2
import importlib
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin

try:
    import json
    import torch
    import trt_pose.coco
    import trt_pose.models
    from torch2trt import TRTModule
    import torchvision.transforms as transforms
    from trt_pose.parse_objects import ParseObjects
except:
    pass

class PoseExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, model_path='./models/pose.tflite'):
        self.model_path = model_path
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        _, self.input_dim, _, _ = self.input_details[0]['shape']
        _, self.mp_dim, _, self.ky_pt_num = self.output_details[0]['shape']
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        '''
        X is an iterable listing image paths or numpy arrays
        '''
        feat_array = []
        filepath = True if isinstance(X[0], str) else False
        for row in X:

            # Read image and resize for model
            image = Image.open(row) if filepath else Image.fromarray(row)

            image = image.resize((self.input_details[0]['shape'][1],self.input_details[0]['shape'][2]), Image.NEAREST)
            image = np.expand_dims(np.asarray(image).astype(self.input_details[0]['dtype'])[:, :, :3], axis=0)

            # Run inference
            self.interpreter.set_tensor(self.input_details[0]['index'], image)
            self.interpreter.invoke()
            result = self.interpreter.get_tensor(self.output_details[0]['index'])

            # Process result and create feature array
            res = result.reshape(1, self.mp_dim**2, self.ky_pt_num)
            max_idxs = np.argmax(res, axis=1)
            coords = list(map(lambda x: divmod(x, self.mp_dim), max_idxs))
            feature_vec = np.vstack(coords).T.reshape(2 * self.ky_pt_num, 1)
            feat_array.append(feature_vec)

        return np.array(feat_array).squeeze()

class TRTPoseExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, model_path='./models/resnet18_baseline_att_224x224_A_epoch_249_trt.pth'):
        self.model_path = model_path
        with open('./models/human_pose.json', 'r') as f:
            self.human_pose = json.load(f)
        self.topology = trt_pose.coco.coco_category_to_topology(self.human_pose)
        self.model_trt = TRTModule()
        self.model_trt.load_state_dict(torch.load(self.model_path))
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
        self.std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
        self.device = torch.device('cuda')
        self.parse_objects = ParseObjects(self.topology)
        self.get_keypoints = GetKeypoints(self.topology)

    def preprocess(self, image):
        global device
        device = torch.device('cuda')
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = transforms.functional.to_tensor(image).to(device)
        image.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
        return image[None, ...]
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        feat_array = []
        filepath = True if isinstance(X[0], str) else False
        for row in X:

            # Read image and resize for model
            image = cv2.imread(row) if filepath else row
            data = self.preprocess(image)
            cmap, paf = self.model_trt(data)
            cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
            counts, objects, peaks = self.parse_objects(cmap, paf) 
            feature_vec = self.get_keypoints(image, counts, objects, peaks)
            feat_array.append(feature_vec)

        return np.array(feat_array).squeeze()

class GetKeypoints(object):
    def __init__(self, topology):
        self.topology = topology
        self.body_labels = {0:'nose', 1: 'lEye', 2: 'rEye', 3:'lEar', 4:'rEar', 5:'lShoulder', 6:'rShoulder',
               7:'lElbow', 8:'rElbow', 9:'lWrist', 10:'rWrist', 11:'lHip', 12:'rHip', 13:'lKnee', 14:'rKnee',
              15:'lAnkle', 16:'rAnkle', 17:'neck'}
        self.body_parts = sorted(self.body_labels.values())

    def __call__(self, image, object_counts, objects, normalized_peaks):
        topology = self.topology
        height = image.shape[0]
        width = image.shape[1]

        K = topology.shape[0]
        count = int(object_counts[0])
        if count > 1:
            count = 1
        K = topology.shape[0]
        
        body_dict = {}
        feature_vec = []
        for i in range(count):
            obj = objects[0][i]
            C = obj.shape[0]
            for j in range(C):
                k = int(obj[j])
                if k >= 0:
                    peak = normalized_peaks[0][j][k]
                    x = round(float(peak[1]) * width)
                    y = round(float(peak[0]) * height)
                    body_dict[self.body_labels[j]] = [x,y]
        for part in self.body_parts:
            feature_vec.append(body_dict.get(part, [0,0]))
        feature_vec = [item for sublist in feature_vec for item in sublist]
        return feature_vec

if __name__ == '__main__':
    extractor = PoseExtractor()
    image = cv2.imread('/your/sample/image.jpg')
    sample = extractor.transform([image])
    print(type(sample))
    print(sample.shape)
    print(sample)
