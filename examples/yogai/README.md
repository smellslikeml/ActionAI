# YogAI 

YogAI is a responsive virtual yoga instructor using pose estimation to guide and correct a yogi that runs on a raspberry pi smart mirror. 

#### Read More:
- [MagPi Article](https://www.raspberrypi.org/blog/yoga-training-with-yogai-and-a-raspberry-pi-smart-mirror-the-magpi-issue-80/)
- [Hackster writeup](https://www.hackster.io/yogai/yogai-smart-personal-trainer-f53744)

## Dependencies

You'll need have the following installed:
- python3
- tensorflow 1.11 -  pip wheel for 3.5 w/tflite working thanks to [PINTO0309](https://github.com/PINTO0309/Tensorflow-bin)
- opencv3
- sci-kit learn

### Hardware
- raspberry pi 3+
- webcam
- speaker with aux 
- monitor
- one way mirror + framing materials

## Install
```
$ git clone https://www.github.com/smellslikeml/YogAI
$ cd YogAI
$ ./install.sh 
```

## Model
We're using a tflite Convolutional Pose Machine (CPM) model we found [here](https://github.com/edvardHua/PoseEstimationForMobile/tree/master/release/cpm_model). The table below offers more information about the model we are running for labeling and inference.

| Model | Input shape | Output shape | Model size | Inference time (rpi3) |
| --- | --- | --- | --- | --- |
| [CPM](https://arxiv.org/pdf/1602.00134.pdf) | ``` [1, 192, 192, 3] ``` | ``` [1, 96, 96, 14] ``` | 2.6 MB | ~2.56 FPS |

Using this model and the ``` label.py ``` script on yoga sample poses will output 28 dim arrays of body part coordinates into a csv file.

## Training 

The [Hackster post](https://www.hackster.io/yogai/yogai-smart-personal-trainer-f53744) will show you how to obtain training samples for your desired poses. Use the ```label.py``` script to transform the images into 28 dim arrays with labels. The knn.ipynb is a jupyter notebook to help you train a KNN to classify yoga poses. You want to make sure your samples follow this directory structure:

```
├── poses
│   ├── plank
│   │   ├── sample1.jpg
│   │   ├── sample2.jpg
│   │   ├── ...
│   ├── cow
│   │   ├── sample1.jpg
│   │   ├── sample2.jpg
│   │   ├── ...
.   .
.   .
```

## Run
After you've trained the classifier on your samples, you should have a pickled model in the ``` ./models ``` directory. Simply run
```
python3 app.py
```
to get your YogAI instructor running!

## References
[1] Convolutional Pose Machine : https://arxiv.org/pdf/1602.00134.pdf

[2] Tensorflow wheels w/ tflite : https://github.com/PINTO0309/Tensorflow-bin

[3] Pose estimation for mobile : https://github.com/edvardHua/PoseEstimationForMobile

[4] Pose estimation tensorflow implementation : https://github.com/ildoonet/tf-pose-estimation
