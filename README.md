# ActionAI 🤸

[![Python 3.x](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/downloads/release/python-370/) [![Join the chat at https://gitter.im/action-ai/community](https://badges.gitter.im/action-ai/community.svg)](https://gitter.im/action-ai/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
![stars](https://img.shields.io/github/stars/smellslikeml/ActionAI)
![forks](https://img.shields.io/github/forks/smellslikeml/ActionAI)
![license](https://img.shields.io/github/license/smellslikeml/ActionAI)
![twitter](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Fgithub.com%2Fsmellslikeml%2FActionAI)

ActionAI is a python library for training machine learning models to classify human action. It is a generalization of our [yoga smart personal trainer](https://www.hackster.io/yogai/yogai-smart-personal-trainer-f53744), which is included in this repo as an example.

> 📈 **#3 on [GitHub Trending](https://github.com/larsbijl/trending_archive/blob/master/2020/2020-01/2020-01-20.md)** (Python, Jan 2020) · 🟩 featured in [NVIDIA Jetson Community Projects](https://developer.nvidia.com/embedded/community/jetson-projects/actionai) · 📄 cited in research

<p align="center">
  <img src="https://github.com/smellslikeml/ActionAI/blob/master/assets/ActionAI_main.gif">
</p>

## Getting Started 
These instructions will show how to prepare your image data, train a model, and deploy the model to classify human action from image samples. See deployment for notes on how to deploy the project on a live stream.

### Installation

Add the smellslikeml PPA and install with the following:
```
sudo add-apt-repository ppa:smellslikeml/ppa
sudo apt update

# Install with:
sudo apt-get install actionai
```

Make sure to configure the working directory with:
```
actionai configure
```

### Using the CLI

Organize your training data in subdirectories like the example below. The `actionai` cli will automatically create a dataset from subdirectories of videos where each subdirectory is a category label.

```
.
└── dataset/
    ├── category_1/
    │   └── *.mp4
    ├── category_2/
    │   └── *.mp4
    ├── category_3/
    │   └── *.mp4
    └── ...
```

Then you can train a model with:
```
actionai train --data=/path/to/your/data/dir --model=/path/to/your/model/dir
```

And then run inference on a video with:
```
actionai predict --model=/path/to/your/model/dir --video=/path/to/your/video.mp4
```

View the default `config.ini` file included in this branch for additional configurations. You can pass your own config file using the `--cfg` flag.


## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE.md](LICENSE.md) file for details

## References

* [Human activity recognition-based path planning for autonomous vehicles (SIViP 2021)](https://link.springer.com/article/10.1007/s11760-020-01800-6) — cites ActionAI
* [Muscle Vision: Keypoint-Based Pose Classification (2022)](https://arxiv.org/abs/2203.12111) — builds on the YogAI approach
* [Hackster post](https://www.hackster.io/yogai/yogai-smart-personal-trainer-f53744)
* [YogAI article](https://www.raspberrypi.org/blog/yoga-training-with-yogai-and-a-raspberry-pi-smart-mirror-the-magpi-issue-80/)
* [Convolutional Pose Machine](https://arxiv.org/pdf/1602.00134.pdf)
* [Pose estimation for mobile](https://github.com/edvardHua/PoseEstimationForMobile)
* [Pose estimation tensorflow implementation](https://github.com/ildoonet/tf-pose-estimation)

## Citation

If you use ActionAI in your work, please cite the repository:

```bibtex
@misc{actionai2019,
  author = {Rodriguez, Terry and Mayorquin, Salma},
  title  = {ActionAI: Real-Time Spatio-Temporally Localized Activity Detection by Tracking Body Keypoints},
  year   = {2019},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/smellslikeml/ActionAI}}
}
```
