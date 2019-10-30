#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
import random

def kp_rotate(vec, shift=0.2):
    v = vec.copy()
    v -= [46,46]
    N = np.linalg.norm(v)
    n_theta = np.arctan(v[1] / v[0]) - shift
    return np.array([int(np.cos(n_theta)* N), int(np.sin(n_theta)* N)])

def rand_shift(vec, v):
    return vec + v * np.random.randint(-10, 10)

def vert_swap(vec, x_coords, y_coords, N=96):
    def col_swap(col):
        return (N - col) % 96
    return (vec * x_coords) + apply_vec(col_swap, (vec * y_coords))

@np.vectorize
def apply_vec(f,x):
    return f(x)


def augment(df, x_coords, y_coords, rows=1000):
    lst = []
    for row in range(rows):
        x, y= df.iloc[row, :-1], df.iloc[row,-1]
        for idx in range(5):
            v = random.choice([x_coords, y_coords]) #randomly choose an index to shift
            ss = round(random.uniform(-0.2,0.2),2)
            vec = rand_shift(x, v)
            rot_vec = np.apply_along_axis(kp_rotate, 1 , x.reshape([N,2]), shift=ss).flatten()
            lst.append(list(vec) + [y])
            lst.append(list(rot_vec) + [y])
        
        if random.choice([True, False]):
            #Vertical flip
            v_vec = vert_swap(x, x_coords, y_coords)
            lst.append(list(v_vec) + [y])
    return pd.DataFrame(lst)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data Augmentor')
    parser.add_argument('--folder', type=str, default='../data/yoga/poses.csv')
    parser.add_argument('--out_file', type=str, default='../data/yoga/augmented_poses.csv')
    parser.add_argument('--array_dim', type=int, default=28)
    args = parser.parse_args()

    #Variable for array_dim /2
    N = args.array_dim / 2

    #Reading poses
    df = pd.read_csv('./data/yoga/poses.csv', header=None)

    x_coords = np.array([1,0] * N, dtype=np.int32)       # get x coordinates only
    y_coords = np.flip(x_coords, axis=0)   # get y coordinates only


    out_df = augment(df, x_coords, y_coords)
    out_df.to_csv('./data/yoga/augmented_poses.csv')
