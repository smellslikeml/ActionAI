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

echo 'Done!'
