#!/usr/bin/env python3
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from multiprocessing import cpu_count


dataset = pd.read_csv('./data/yoga/augmented_poses.csv',  index_col=0)
dataset.head()

# Spliting the points and the labels
X = dataset.iloc[:, :-1].values  
y = dataset.iloc[:, 28].values

# And split the data into appropriate data sets

class_names = list(set(y))
num_class = len(class_names)
cores = cpu_count()

clf = XGBClassifier(max_depth=6, 
                    learning_rate=0.01, 
                    n_estimators=300, 
                    objective='multi:softmax', 
                    n_jobs=cores, 
                    num_class=num_class)

#preds = clf.fit(X_train, y_train).predict(X_test)
clf.fit(X, y)
filename = './models/yoga_poses.sav'
pickle.dump(clf, open(filename, 'wb'))

