#!/usr/bin/env python3

import pandas as pd
import numpy as np
import random

#Reading poses
df = pd.read_csv('./data/yoga/poses.csv', header=None)

x_coords = np.array([1,0] * 14, dtype=np.int32)       # get x coordinates only
y_coords = np.flip(x_coords, axis=0)   # get y coordinates only

def kp_rotate(vec, shift=0.2):
    v = vec.copy()
    v -= [46,46]
    N = np.linalg.norm(v)
    n_theta = np.arctan(v[1] / v[0]) - shift
    return np.array([int(np.cos(n_theta)* N), int(np.sin(n_theta)* N)])

def rand_shift(vec, v):
    return vec + v * np.random.randint(-10, 10)

def vert_swap(vec, N=96):
    def col_swap(col):
        return (N - col) % 96
    return (vec * x_coords) + apply_vec(col_swap, (vec * y_coords))

@np.vectorize
def apply_vec(f,x):
    return f(x)

out_lst = []

for row in range(df.shape[0]):
    x, y= df.iloc[row, :-1], df.iloc[row,-1]
    for idx in range(5):
        v = random.choice([x_coords, y_coords]) #randomly choose an index to shift
        ss = round(random.uniform(0,1),2)
        vec = rand_shift(x, v)
        rot_vec = np.apply_along_axis(kp_rotate, 1 , x.reshape([14,2]), shift=ss).flatten()
        out_lst.append(list(vec) + [y])
        out_lst.append(list(rot_vec) + [y])

    #Vertical flip
    v_vec = vert_swap(x)
    out_lst.append(list(v_vec) + [y])

out_df = pd.DataFrame(out_lst)
out_df.to_csv('./data/yoga/augmented_poses.csv')
