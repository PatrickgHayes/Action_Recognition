import os
import glob
import subprocess
import config
from tqdm import tqdm
from joblib import Parallel, delayed

### GLOBALS
dataset_dir = '/datasets/home/71/671/cs291dag/MiniKinetics/train/'
###



def resize_crop(img):
    '''
    resize the image frame to a random 224 by 224
    '''

    aspect_ratio = float(img.shape[1]) / float(img.shape[0])
    new_w = 0
    new_h = 0
    if aspect_ratio <= 1.0:
        new_w = 256
        new_h = int(256 / aspect_ratio)
    else:
        new_h = 256
        new_w = int(256 * aspect_ratio)

    random.seed(datetime.now())
    resize = misc.imresize(img, (new_h, new_w), 'bilinear')
    wrange = resize.shape[1] - 224
    hrange = resize.shape[0] - 224
    w_crop = random.randint(0, wrange)
    h_crop = random.randint(0, hrange)

    return resize[h_crop:h_crop+224, w_crop:w_crop+224]


def createJPGs(video, dest):
    '''
    creates the jpegs by calling the ffmpeg
    '''
#    print('here: ' + video)
    dest_name = os.path.splitext(video)[0]
    if dest_name not in os.listdir():
        os.mkdir(dest_name)
    video_path = os.path.join(label_path, video)
    dest_name = os.path.join(dest_name, "img%4d.jpg")

    video = str(video).replace(' ', '\ ')
    dest = str(dest_name).replace(' ', '\ ')
    command = "ffmpeg -i " + video + " -r 25.0 " + dest
    proc = subprocess.Popen(
        command,
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd='.'
    )
    out, err = proc.communicate()
    print(out)
    print(err)

def main():
    '''
    1. move into the video directory
    2. extract the frames and resize them using
        bilinear extrapolation
    3. randomly select a 224 by 224 crop
    4. change pixel values to be [-1, 1]
    '''
    os.chdir(dataset_dir)
    for label in tqdm(os.listdir()):
        if label.startswith("."):
            continue

        print("===================== " + label + " ======================== ")
        label_path = os.path.join(dataset_dir, label)
        os.chdir(label_path)
        files_list = glob.glob("*.mp4")

        Parallel(n_jobs=-1, verbose=True)(delayed(createJPGs)(video, label_path) for video in files_list)


        # for video in glob.glob("*.mp4"):
        #     print("\tprocessing video: " + video)
        #     dest_name = os.path.splitext(video)[0]
        #     if dest_name not in os.listdir():
        #         os.mkdir(dest_name)
        #     video_path = os.path.join(label_path, video)
        #     dest_name = os.path.join(dest_name, "img%4d.jpg")
        #     createJPGs(video_path, dest_name)

if __name__ == '__main__':
    main()
