#Cleaning up image
echo 'Updating...'
sudo apt-get update
sudo apt-get -y dist-upgrade
sudo apt-get install python3-dev
echo

# Installing OpenCV3
echo 'Installing OpenCV'
sudo apt-get install libopencv-dev python3-opencv
echo

echo 'Installing tensorflow + dependencies'
#Install tensorflow with tflite fixed
pip3 install ./utils/tensorflow-1.11.0-cp35-cp35m-linux_armv7l.whl
echo 

echo 'Installing sklearn + xgboost + dependencies'
#Install sci-kit learn, pandas, xgboost
sudo apt-get install gfortran libopenblas-dev liblapack-dev
sudo apt-get install python-pandas
pip3 install scikit-learn
pip3 install xgboost
echo

echo 'Installing keras + dependencies'
#Install keras and dependencies
sudo apt-get install libblas-dev
sudo apt-get install liblapack-dev
sudo apt-get install python3-dev 
sudo apt-get install libatlas-base-dev
sudo apt-get install gfortran
sudo apt-get install python3-setuptools
sudo apt-get install python3-scipy

sudo apt-get update
sudo apt-get install python3-h5py
sudo pip3 install keras

echo 'Done!'
