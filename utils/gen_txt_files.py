import os
import glob


def main():
    DATA_DIR = "/Users/dewalgupta/Documents/ucsd/291d/activitynet/data/"
    OUTPUT_DIR = "/Users/dewalgupta/Documents/ucsd/291d/activitynet/Action_Recognition/"
    os.chdir(DATA_DIR)

    categories = []
    with open(os.path.join(OUTPUT_DIR, "videos.txt"), 'w') as f:
        for label in os.listdir():
            if label.startswith('.'):
                continue

            categories.append(label)
            for video in glob.glob(os.path.join(label, "*.mp4")):
                filepath = os.path.splitext(video)[0]
                f.write(os.path.join(DATA_DIR, filepath) + "\n")

    with open(os.path.join(OUTPUT_DIR, "label_map.txt"), 'w') as f:
        for cat in categories:
            f.write(cat + "\n")

if __name__ == '__main__':
    main()
