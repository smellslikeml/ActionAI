import os
import pandas as pd

if __name__ == '__main__':
    import argparse
    import importlib
    parser = argparse.ArgumentParser(description='Generate csv from image/video data')
    parser.add_argument('--config', type=str, default='conf',
                        help="name of config .py file inside config/ directory, default: 'conf'")
    args = parser.parse_args()
    config = importlib.import_module('config.' + args.config)

    labeled_images = []
    for subdir, dirs, files in os.walk(config.images_dir):
        for img in files:
            image_path = os.path.join(subdir, img)
            label = subdir.split('/')[-1]
            labeled_images.append([image_path, label])
    df = pd.DataFrame(labeled_images, columns=['image', 'label'])
    df.to_csv(config.csv_path, encoding='utf-8', index=False)
