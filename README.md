# ActionAI 🤸

[![Python 3.x](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/downloads/release/python-370/)
![stars](https://img.shields.io/github/stars/smellslikeml/ActionAI)
![forks](https://img.shields.io/github/forks/smellslikeml/ActionAI)
![license](https://img.shields.io/github/license/smellslikeml/ActionAI)
![twitter](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Fgithub.com%2Fsmellslikeml%2FActionAI)

ActionAI is a python library for training machine learning models to classify human action. It is a generalization of our [yoga smart personal trainer](https://www.hackster.io/yogai/yogai-smart-personal-trainer-f53744), which is included in this repo as an example.

<p align="center">
  <img src="https://github.com/smellslikeml/ActionAI/blob/master/assets/ActionAI_main.gif">
</p>

## Getting Started 
These instructions will show how to prepare your image data, train a model, and deploy the model to classify human action from image samples. See deployment for notes on how to deploy the project on a live stream.

### Prerequisites
- [tensorflow 2.0](https://www.tensorflow.org)
- [scikit-learn](https://scikit-learn.org/stable/)
- [opencv](https://opencv-python-tutroals.readthedocs.io/en/latest/)
- [pandas](https://pandas.pydata.org)
- [pillow](https://pillow.readthedocs.io/en/stable/)

### Installing
We recommend using a virtual environment to avoid any conflicts with your system's global configuration. You can install the required dependencies via pip:

### Jetson Nano Installation
We use the [trt_pose repo](https://github.com/NVIDIA-AI-IOT/trt_pose) to extract pose estimations. Please look to this repo to install the required dependencies. 
You will also need to download these zipped [model assets](https://drive.google.com/open?id=1SkPn4vzZofCtwReodtAsnwYgVkONR5-G) and unzip the package into the ```models/``` directory. 

```bash
# Assuming your python path points to python 3.x 
$ pip install -r requirements.txt
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
$ python preprocess.py
```
This will stage the labeled image dataset in a csv file written to the ```data/``` directory.

## Training
After reading the csv file into a dataframe, a custom scikit-learn transformer estimates body keypoints to produce a low-dimensional feature vector for each sample image. This representation is fed into a scikit-learn classifier set in the config file. This approach works well for lightweight applications that require classifying a pose like the [YogAI](https://www.hackster.io/yogai/yogai-smart-personal-trainer-f53744) usecase:

<p align="center">
  <img src="https://github.com/smellslikeml/ActionAI/blob/master/assets/actionai_example.gif">
</p>


Run the train.py script to train and save a classifier
```bash
$ python train.py
```

The pickled model will be saved in the ```models/``` directory


<p align="center">
  <img src="https://github.com/smellslikeml/ActionAI/blob/master/assets/yogai_squat_or_not.gif">
</p>

To train a more complex model to classify a sequence of poses culminating in an action (ie. squat or spin), use the ```train_sequential.py``` script. This script will train an LSTM model to classify movements.

```bash
$ python train_sequential.py
```


## Deployment
We've provided a sample inference script, ```inference.py```, that will read input from a webcam, mp4, or rstp stream, run inference on each frame, and print inference results. 

If you are running on a Jetson Nano, you can try running the ```iva.py``` script, which will perform multi-person tracking and activity recognition like the demo gif above *Getting Started*. Make sure you have followed the Jetson Nano installation instructions above and simply run:
```bash
$ python iva.py 0

# or if you have a video file

$ python iva.py /path/to/file.mp4
```
If specified, this script will write a labeled video as ```out.mp4```. This demo uses a sample model called ```lstm_spin_squat.h5``` to classify spinning vs. squatting. Change the model and motion dictionary under the ```RUNSECONDARY``` flag to run your own classifier. 

### Teachable Machine
<p align="center">
  <img src="https://github.com/smellslikeml/ActionAI/blob/master/assets/teachable.gif">
</p>

We've also included a script under the experimental folder, ```teachable_machine.py```, that supports labelling samples via a PS3 Controller on a Jetson Nano and training in real-time from a webcam stream. This will require these extra dependencies:
* [Pygame](https://www.pygame.org/docs/ref/joystick.html)
* [PS3 Controller](https://docs.donkeycar.com/parts/controllers/#ps3-controller)

To test it, run:
``` bash
# Using a webcam
$ python experimental/teachable_machine.py /dev/video0  

# Using a video asset
$ python experimental/teachable_machine.py /path/to/file.mp4  
```
This script will also write labelled data into a csv file stored in ```data/``` directory and produce a video asset ```out.mp4```. 

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
