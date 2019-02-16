#!/usr/bin/env python
import requests
from bs4 import BeautifulSoup

urls = ['https://www.yogajournal.com/poses/types/arm-balances', 'https://www.yogajournal.com/poses/types/balancing', 'https://www.yogajournal.com/poses/types/binds', 'https://www.yogajournal.com/poses/types/chest-openers', 'https://www.yogajournal.com/poses/types/core', 'https://www.yogajournal.com/poses/types/forward-bends', 'https://www.yogajournal.com/poses/types/hip-openers', 'https://www.yogajournal.com/poses/types/inversions', 'https://www.yogajournal.com/poses/types/seated-twists', 'https://www.yogajournal.com/poses/types/pranayama', 'https://www.yogajournal.com/poses/types/restorative', 'https://www.yogajournal.com/poses/types/standing', 'https://www.yogajournal.com/poses/types/strength', 'https://www.yogajournal.com/poses/types/twists', 'https://www.yogajournal.com/poses/types/backbends', 'https://www.yogajournal.com/poses/types/bandha']


def main():
    pose_dict = {}
    for url in urls:
        res = requests.get(url)
        soup = BeautifulSoup(res.text)


        for dd in soup.find_all('div'):
            sansk = dd.find('p')
            eng = dd.find('h2')
            try:
                pose_dict[sansk.text] = eng.text
            except:
                pass
    return pose_dict


print(main())
