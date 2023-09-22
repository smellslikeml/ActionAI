import os
import glob
import json
import subprocess
import numpy as np
import pandas as pd

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split


def train(data_dir, model_dir, window_size, learning_rate, epochs, batch_size):
    # Step 1: Extract keypoints using DLStreamer and save to JSON
    labels = os.listdir(data_dir)
    label_mapping = {label: idx for idx, label in enumerate(labels)}

    for label in labels:
        label_dir = os.path.join(data_dir, label)
        for mp4_file in glob.glob(f"{label_dir}/*.mp4"):
            output_json = mp4_file.replace('.mp4', '.json')
            gst_command = f"""gst-launch-1.0 filesrc location={mp4_file} ! decodebin ! gvaclassify model=/home/dlstreamer/intel/dl_streamer/models/intel/human-pose-estimation-0001/FP32/human-pose-estimation-0001.xml model-proc=/opt/intel/dlstreamer/samples/gstreamer/gst_launch/human_pose_estimation/model_proc/human-pose-estimation-0001.json device=CPU inference-region=full-frame ! queue ! gvametaconvert add-tensor-data=true ! gvametapublish file-format=json-lines file-path={output_json} ! fakesink async=false """
            subprocess.run(gst_command, shell=True)

    # Load JSON files as training data
    X_data = []
    y_data = []

    labels = os.listdir(data_dir)
    label_mapping = {label: idx for idx, label in enumerate(labels)}

    for label in labels:
        label_dir = os.path.join(data_dir, label)
        for mp4_file in glob.glob(f"{label_dir}/*.mp4"):
            output_json = mp4_file.replace('.mp4', '.json')

            # Load JSON file into DataFrame
            df = pd.read_json(output_json, lines=True)

            sequences = []
            for idx, row in df.iterrows():
                for obj in row['objects']:
                    keypoints = obj['tensors'][0]['data']
                    keypoints = np.array(keypoints) / np.linalg.norm(keypoints)
                    sequences.append(keypoints)

                    if len(sequences) == window_size:
                        X_data.append(sequences)
                        y_data.append(label_mapping[label])
                        sequences = sequences[1:]

    # Convert to numpy arrays and reshape for LSTM training
    X_data = np.array(X_data)
    y_data = np.array(y_data)

    X_data = np.reshape(X_data, (-1, window_size, 36))
    y_data = y_data.reshape(-1, 1)

    # LSTM model
    model = Sequential([
        LSTM(16, dropout=0.2, recurrent_dropout=0.2, input_shape=(window_size, 36)),
        Dense(16, activation="relu"),
        Dense(len(labels), activation="softmax")
    ])

    opt = Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_data, y_data, epochs=epochs, batch_size=batch_size)

    # Save your model
    model_save_path = os.path.join(model_dir, 'model.h5')
    model.save(model_save_path)
    with open(os.path.join(model_dir, 'labels.txt'), mode='wt', encoding='utf-8') as label_file:
        label_file.write('\n'.join(labels))

    return f"Model saved successfully: {model_save_path}"
