#!/usr/bin/env python
import os
import sys
import requests
import numpy as np
import pandas as pd
from collections import defaultdict
from bs4 import BeautifulSoup

urls = ['https://www.yogajournal.com/poses/types/arm-balances', 'https://www.yogajournal.com/poses/types/balancing', 'https://www.yogajournal.com/poses/types/binds', 'https://www.yogajournal.com/poses/types/chest-openers', 'https://www.yogajournal.com/poses/types/core', 'https://www.yogajournal.com/poses/types/forward-bends', 'https://www.yogajournal.com/poses/types/hip-openers', 'https://www.yogajournal.com/poses/types/inversions', 'https://www.yogajournal.com/poses/types/seated-twists', 'https://www.yogajournal.com/poses/types/pranayama', 'https://www.yogajournal.com/poses/types/restorative', 'https://www.yogajournal.com/poses/types/standing', 'https://www.yogajournal.com/poses/types/strength', 'https://www.yogajournal.com/poses/types/twists', 'https://www.yogajournal.com/poses/types/backbends', 'https://www.yogajournal.com/poses/types/bandha']


def get_data():
    pose_dict = defaultdict(dict)
    for url in urls:
        res = requests.get(url)
        group = url.split('/')[-1]
        soup = BeautifulSoup(res.text)

        for dd in soup.find_all('div'):
            sansk = dd.find('p')
            eng = dd.find('h2')
            try:
                pose_dict[group][sansk.text] = eng.text
            except:
                pass
    return pose_dict


def rand_flow(N=100, lang='eng'):
    for idx in range(N):
        ky = np.random.choice(list(pose_dict.keys()))
        if lang == 'eng':
            vals = np.random.choice(list(pose_dict[ky].keys()))
        else:
            vals = np.random.choice(list(pose_dict[ky].values()))
        yield ky, pose_dict[ky][vals]


if __name__ == '__main__':
    if not os.path.exists('pose_df.csv'):
        pose_dict = get_data()
        #print(pose_dict[list(pose_dict.keys())[0]])
        data_lst = []
        for pose in pose_dict.keys():
            for nm in pose_dict[pose]:
                data_lst.append([pose, nm, pose_dict[pose][nm]])
        df = pd.DataFrame(data_lst)
        df.columns = ['type', 'sanskrit', 'english']
        df.to_csv('pose_df.csv', index=False)
    else:
        df = pd.read_csv('pose_df.csv', index_col=None)
        pose_dict = defaultdict(dict)
        for row in df.iterrows():
            tp, sk, en = row[1].values
            pose_dict[tp][sk] = en
    fl = rand_flow()
    while True:
        try:
            tp, pose = next(fl)
            print(pose)
        except StopIteration:
            sys.exit(0)
