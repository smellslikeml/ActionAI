from setuptools import setup, find_packages

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="actionai",
    version="0.0.1",
    packages=find_packages(),
    package_data={"actionai": ["config.ini"]},
    install_requires=[
        "tensorflow>=2.6.2",
        "scipy",
        "scikit-learn",
        "opencv-contrib-python",
        "pandas",
        "pillow",
        "tqdm",
        ],
    entry_points={
        "console_scripts": [
            "actionai=actionai.cli:main",
        ],
    },
    long_description=long_description,
    long_description_content_type='text/markdown'
)
