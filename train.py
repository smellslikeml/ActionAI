"""
Training - run this script after preprocessing your image data!

If you are on a Jetson Nano, import TRTPoseExtractor from transformer and modify the actionModel function to use this instead. 
ie.

from transformer import PoseExtractor

def actionModel(classifier):
    pipeline = Pipeline([
               ('pose_extractor', TRTPoseExtractor()),
               ...
               ])
    return pipeline
"""

import pandas as pd
from sklearn.pipeline import Pipeline

from transformer import PoseExtractor

def actionModel(classifier):
    pipeline = Pipeline([
               ('pose_extractor', PoseExtractor()),
               ('classifier', classifier)])
    return pipeline

def trainModel(csv_path, pipeline):
    df = pd.read_csv(csv_path)
    X = df['image'].values
    y = df['label']
    pipeline = pipeline.fit(X, y)
    return pipeline.get_params()['steps'][1][1]  

if __name__ == '__main__':
    import pickle
    import argparse
    import importlib

    parser = argparse.ArgumentParser(description='Train pose classifier')
    parser.add_argument('--config', type=str, default='conf',
                        help="name of config .py file inside config/ directory, default: 'conf'")
    args = parser.parse_args()
    config = importlib.import_module('config.' + args.config)

    pipeline = actionModel(config.classifier())
    model = trainModel(config.csv_path, pipeline)

    # Dump the model to file
    pickle.dump(model, open(config.classifier_model, 'wb'), protocol=2)
