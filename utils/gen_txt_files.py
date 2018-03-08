import os
import glob


def main():
    DATA_DIR = "/Users/mithuncd/ActivityNet/MicroKinetics/train"
    OUTPUT_DIR = "/Users/mithuncd/Action_Recognition/config/"
    os.chdir(DATA_DIR)

    categories = []
    with open(os.path.join(OUTPUT_DIR, "micro_kinetics_train.txt"), 'w') as f:
        for label in os.listdir(DATA_DIR):
            if label.startswith('.'):
                continue

            categories.append(label)
            for video in glob.glob(os.path.join(label, "*.mp4")):
                filepath = os.path.splitext(video)[0]
                if os.listdir(os.path.join(DATA_DIR, filepath)):
                    f.write(os.path.join(DATA_DIR, filepath) + "\n")

    with open(os.path.join(OUTPUT_DIR, "label_map_micro_kinetics.txt"), 'w') as f:
        for cat in categories:
            f.write(cat + "\n")


if __name__ == '__main__':
    main()
