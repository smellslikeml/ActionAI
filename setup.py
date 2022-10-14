from setuptools import setup, find_packages

setup(
    name='actionai',
    version='0.1.0',
    description='Real-time spatio-temporally localized activity detection by tracking body keypoints.',
    author='Terry Rodriguez and Salma Mayorquin (smellslikeml)',
    author_email='contact@smellslikeml.com',
    url='https://smellslike.ml',
    packages=find_packages(include=['actionai', 'actionai.*']),
    install_requires=[
        'tensorflow>=2.6.2',
        'scipy',
        'scikit-learn',
        'opencv-contrib-python',
        'pandas',
        'pillow'
    ],
    setup_requires=['pytest-runner', 'flake8'],
    tests_require=['pytest'],
    package_data={'actionai': ['models/pose.tflite']}
)
