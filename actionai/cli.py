import os
import argparse
import configparser
from pkg_resources import resource_filename

from .train import *
from .predict import *

from pprint import pprint

def main():
    parser = argparse.ArgumentParser(description="ActionAI CLI - Train and test video classification models.")
    
    # Create a subparsers object
    subparsers = parser.add_subparsers(dest='command', help='Sub-command help')

    # Define 'train' subaction
    train_parser = subparsers.add_parser("train", help="Train a model on a dataset.")
    train_parser.add_argument("--cfg", required=False, default=None, help="Model configurations")
    train_parser.add_argument("--data", required=True, help="Dataset directory")
    train_parser.add_argument("--model", required=False, default=None, help="Output directory for model assets")
    
    # Define 'predict' action
    predict_parser = subparsers.add_parser("predict", help="Run a trained model on a video.")
    predict_parser.add_argument("--cfg", required=False, default=None, help="Model configurations")
    predict_parser.add_argument("--model", required=False, default=None, help="Output directory for model assets")
    predict_parser.add_argument("--video", required=True, help="Video to predict model on")
    predict_parser.add_argument("--save", required=False, action='store_true', help="Save results to file")
    predict_parser.add_argument("--display", required=False, action='store_true', help="Display results")

    args = parser.parse_args()
    config = configparser.ConfigParser()

    # Locate the default config.ini
    default_config_path = resource_filename('actionai', 'config.ini')
    config.read(default_config_path)

    package_directory = os.path.dirname(os.path.realpath(__file__))
    # Add this directory to PYTHONPATH
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    os.environ['PYTHONPATH'] = f"{package_directory}:{current_pythonpath}"

    if args.command == 'train':
        if args.cfg is not None:
            cfg = config.read(args.cfg)
        data_dir = args.data 
        model_dir = args.model if args.model is not None else config.get('DEFAULT', 'model_dir', fallback=os.path.expanduser("~"))
        window_size = int(config["DEFAULT"]["window_size"])
        learning_rate = float(config["TRAIN"]["learning_rate"])
        epochs = int(config["TRAIN"]["epochs"])
        batch_size = int(config["TRAIN"]["batch_size"])

        result = train(data_dir, model_dir, window_size, learning_rate, epochs, batch_size)
        pprint(result) 
    elif args.command == 'predict':
        if args.cfg is not None:
            cfg = config.read(args.cfg)
        model_dir = args.model if args.model is not None else config.get('DEFAULT', 'model_dir', fallback=os.path.expanduser("~"))
        window_size = int(config["DEFAULT"]["window_size"])
        video = args.video
        save = args.save
        display = args.display

        print(f"Debug: model_dir = {model_dir}, window_size = {window_size}")

        result = predict(model_dir, window_size, video, save, display)

        pprint(result) 
    else:
        print("Invalid action")

if __name__ == "__main__":
    main()
