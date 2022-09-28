# Training

All preprocessing, training, and deployment configuration variables are stored in the ```conf.py``` file in the ```config/``` directory. You can create your own conf.py files and store them in this directory for fast experimentation.

The ```pipleline.conf.py``` configures a LogisticRegression model as a lightweight classifier.

## Example
After running the ```preprocess.py``` script, you can use ```actionModel()```, to build scikit-learn pipeline before using the ```trainModel()``` method to train:

```python
### Stage your model
pipeline = actionModel(config.classifier())

### Train your model
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
