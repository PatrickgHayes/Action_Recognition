import os
import glob
import random
from configparser import ConfigParser, ExtendedInterpolation

config = ConfigParser(interpolation=ExtendedInterpolation())
config.read("../config/config.ini")

def main():
    DATA_DIR = config['paths']['train_data']
    OUTPUT_DIR = config['paths']['base_fp'] + '/config/'
    os.chdir(DATA_DIR)

    categories = []
    all_vids = []
    for label in os.listdir(DATA_DIR):
        if label.startswith('.'):
            continue

        categories.append(label)
        for video in glob.glob(os.path.join(label, "*.mp4")):
            filepath = os.path.splitext(video)[0]
            all_vids.append(filepath)

    random.shuffle(all_vids)
    with open(os.path.join(OUTPUT_DIR, "train_test.txt"), 'w') as f:
        for path in all_vids:
            if os.listdir(os.path.join(DATA_DIR, path)):
                f.write(os.path.join(DATA_DIR, path) + "\n")

    with open(os.path.join(OUTPUT_DIR, "label_map_test.txt"), 'w') as f:
        for cat in categories:
            f.write(cat + "\n")


if __name__ == '__main__':
    main()
