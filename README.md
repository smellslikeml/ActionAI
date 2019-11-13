# ActionAI 🤸
[![Python 3.x](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/downloads/release/python-370/)
![stars](https://img.shields.io/github/stars/smellslikeml/ActionAI)
![forks](https://img.shields.io/github/forks/smellslikeml/ActionAI)
![license](https://img.shields.io/github/license/smellslikeml/ActionAI)
![twitter](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Fgithub.com%2Fsmellslikeml%2FActionAI)

ActionAI is a python library for training machine learning models to classify human action. It is a generalization of our [yoga smart personal trainer](https://www.hackster.io/yogai/yogai-smart-personal-trainer-f53744), which is included in this repo as an example.

## Getting Started 
These instructions will show how to prepare your image data, train a model, and deploy the model to classify human action from image samples. See deployment for notes on how to deploy the project on a live stream.

### Prerequisites
- [tensorflow 2.0](https://www.tensorflow.org)
- [scikit-learn](https://scikit-learn.org/stable/)
- [opencv](https://opencv-python-tutroals.readthedocs.io/en/latest/)
- [pandas](https://pandas.pydata.org)
- [pillow](https://pillow.readthedocs.io/en/stable/)

### Installing
We recommend using a virtual environment to avoid any conflicts with your systems global configuration. You can install the required dependencies via pip:

```bash
# Assuming your python path points to python 3.x 
pip install -r requirements.txt
```

All preprocessing, training, and deployment configuration variables are stored in the ```conf.py``` file in the ```config/``` directory. You can create your own conf.py files and store them in this directory for fast experimentation.

The ```conf.py``` file included imports a LinearRegression model as our classifier by default.

### Example
After proprocessing your image data using the ```preprocess.py``` script, you can create a model by calling the ```actionModel()```function, which creates a scikit-learn pipeline. Then, call the ```trainModel()``` function with your data to train:

```python
# Stage your model
pipeline = actionModel(config.classifier())

# Train your model
model = trainModel(config.csv_path, pipeline)
```

## Data processing 
Arrange your image data as a directory of subdirectories, each subdirectory named as a label for the images contained in it. Your directory structure should look like this:

```bash
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

To generate a dataset from your images, run the ```preprocess.py``` script.
```bash
python preprocess.py
```
This will stage the labeled image dataset in a csv file written to the ```data/``` directory.

## Training
After reading the csv file into a dataframe, a custom scikit-learn transformer estimates body keypoints to produce a low-dimensional feature vector for each sample image. This representation is fed into a scikit-learn classifier set in the config file. 

Run the train.py script to train and save a classifier
```bash
python train.py
```

The pickled model will be saved in the ```models/``` directory

## Deployment
We've provided a sample inference script, ```inference.py```, that will read input from a webcam, mp4, or rstp stream, run inference on each frame, and print inference results. 

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE.md](LICENSE.md) file for details

## References

* [Hackster post](https://www.hackster.io/yogai/yogai-smart-personal-trainer-f53744)
* [YogAI article](https://www.raspberrypi.org/blog/yoga-training-with-yogai-and-a-raspberry-pi-smart-mirror-the-magpi-issue-80/)
* [Convolutional Pose Machine](https://arxiv.org/pdf/1602.00134.pdf)
* [Pose estimation for mobile](https://github.com/edvardHua/PoseEstimationForMobile)
* [Pose estimation tensorflow implementation](https://github.com/ildoonet/tf-pose-estimation)
