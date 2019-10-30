import os
import argparse
import importlib
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin

parser = argparse.ArgumentParser(description='Transform image data to pose arrays')
parser.add_argument('--config', type=str, default='conf',
                    help="name of config .py file inside config/ directory, default: 'conf'")
args = parser.parse_args()
config = importlib.import_module('config.' + args.config)

class PoseExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, model_path=config.pose_model):
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

