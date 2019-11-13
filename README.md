# ActionAI
[![twitter](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Fgithub.com%2Fsmellslikeml%2FActionAI)](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Fgithub.com%2Fsmellslikeml%2FActionAI)

ActionAI is a python library for training machine learning models to classify human action. It is a generalization of our [yoga smart personal trainer](https://www.hackster.io/yogai/yogai-smart-personal-trainer-f53744), which is included in this repo as an example.

## Dependencies
- tensorflow 2.0
- scikit-learn
- opencv
- pandas
- pillow


## Data Prep 
Arrange your image data as a directory of subdirectories, each subdirectory named as a label for the images contained in it. Your directory structure should look like this:

```
├── images_dir
│   ├── class_1
│   │   ├── sample1.png
│   │   ├── sample2.jpg
│   │   ├── ...
│   ├── class_2
│   │   ├── sample1.png
│   │   ├── sample2.jpg
│   │   ├── ...
.   .
.   .
```
Samples should be standard image files recognized by the pillow library.

To generate a dataset from your images, run the data_generator.py script.
```
python data_generator.py
```
This will stage the labeled image dataset in a csv file written to the ```data/``` directory.

## Training
After reading the csv file into a dataframe, a custom scikit-learn transformer estimates body keypoints to produce a low-dimensional feature vector for each sample image. This representation is fed into a scikit-learn classifier set in the config file. 

Run the train.py script to train and save a classifier
```
python train.py
```

The pickled model will be saved in the ```models/``` directory

## Run
We've provided a sample inference script, ```inference.py```, that will read input from a webcam, mp4, or rstp stream, run inference on each frame, and print inference results. 

