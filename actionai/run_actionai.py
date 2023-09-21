from collections import deque
import tensorflow as tf
import numpy as np
from gstgva import VideoFrame

class ActionAI:
    def __init__(self, model_dir, window_size):
        self.pose_list = deque(maxlen=int(window_size))
        self.model_dir = model_dir
        self.window_size = int(window_size)
        self.model = tf.keras.models.load_model(f"{self.model_dir}/model.h5")
        print(f"Model loaded from {self.model_dir}")
        print(f"Window size is {self.window_size}")

    def add_pose(self, frame: VideoFrame) -> bool:
        for tensor in frame.tensors():
            data = tensor.data()
            self.pose_list.append(data)

            # If pose_list has reached its capacity, make a prediction
            if len(self.pose_list) == self.window_size:
                pose_array = np.array(self.pose_list).reshape(1, self.window_size, 36)
                prediction = np.argmax(self.model.predict(pose_array))
                print(f"Model prediction: {prediction}")

        return True

