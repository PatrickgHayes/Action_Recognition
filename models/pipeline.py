import tensorflow as tf
import os, random
import numpy as np
from scipy import misc
from datetime import datetime


NUM_FRAMES = 64
CROP_SIZE = 224
BATCH_SIZE = 3
STRIDE = NUM_FRAMES
DATA_DIR = "/Users/dewalgupta/Documents/ucsd/291d/activitynet/data"
CLS_DICT_FP = "/Users/dewalgupta/Documents/ucsd/291d/activitynet/Action_Recognition/config/label_map_2.txt"


class Pipeline(object):
    def __init__(self, filepaths):
        self.num_frames = NUM_FRAMES
        self.videos = []
        self._build_cls_dict()
        with open(filepaths, 'r') as f:
            for path in f.readlines():
                path = path.strip()
                if path:
                    self.videos.append(path.strip())

    def _build_cls_dict(self):
        self.cls_dict = {}
        with open(CLS_DICT_FP, 'r') as f:
            for ind, line in enumerate(f.readlines()):
                cls_name = line.strip()
                self.cls_dict[cls_name.lower()] = int(ind)

    def __iter__(self):
        for video_fp in self.videos:
            rgb, label = self._parse(video_fp)
            #         print(label)
            yield (rgb, label)

    def get_frames(self, video_path):
        # np.random.shuffle(videos)  # random shuffle every epoch
        video_path = video_path.strip()
        paths = os.path.split(video_path)
        cls_name = os.path.split(paths[0])[1]

        sorted_list = np.sort(os.listdir(video_path))
        imgs = [os.path.join(video_path, img) for img in sorted_list if img.startswith('img')]
        if self.num_frames <= len(imgs):
            begin = np.random.randint(0, len(imgs) - self.num_frames + 1)
        else:
            begin = 0
            ori_len = len(imgs)
            while len(imgs) < self.num_frames:
                for i in range(0, ori_len, self.stride):
                    imgs.append(imgs[i])
                    if len(imgs) == self.num_frames:
                        break

        imgs_out = imgs[begin:begin + self.num_frames]
        return imgs_out, self.cls_dict[cls_name.lower()]

    def resize_crop(self, img: np.ndarray) -> np.ndarray:
        '''
        resize the image frame to a random 224 by 224
        '''

        aspect_ratio = float(img.shape[1]) / float(img.shape[0])
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

        return resize[h_crop:h_crop + 224, w_crop:w_crop + 224]

    def _parse(self, filename):
        frames, label = self.get_frames(filename)
        rgb = []
        for f in frames:
            npimg = misc.imread(f)
            cropped = self.resize_crop(npimg)
            scaled = 2 * (cropped / 255) - 1
            rgb.append(scaled)

        output_rgb = np.array(rgb)
        return output_rgb, label

    def get_dataset(self):
        return tf.data.Dataset.from_generator(self.__iter__,
                                              (tf.float32, tf.int32),
                                              (tf.TensorShape([NUM_FRAMES, CROP_SIZE, CROP_SIZE, 3]),
                                               tf.TensorShape([])))


if __name__ == '__main__':
    ### FOR TESTING PURPOSES:
    '''
    There are two methods below that this one class can implement. By creating an 
    object of type Pipeline, we can turn it into an iterable, or we can get a
    tensorflow Dataset object from it. Both the examples below accomplish the same
    thing and print (almost) the same things.
    
    NOTE: I created a videos_2.txt so I could test this out with a shorter dataset
    '''
    BATCH_SIZE = 1
    FEATURE_SHAPE = (NUM_FRAMES, CROP_SIZE, CROP_SIZE, 3)

    pipeline = Pipeline("../config/videos_2.txt")

    ################################
    '''
    METHOD 1: USING FEED_DICT
    This method we directly insert the data into the graph. It is 
    supposed to be the fastest and most efficient method (supposedly).
    However, it does not give us any high level APIs for batching and/or
    repeating our input datasets. 
    
    Also NOTE: this is just another way to define the graph. This graph
    is really simple as it just consists of two "place-holders" that we supply
    when we run the session.
    '''

    # iterator = iter(pipeline)
    # with tf.Graph().as_default():
    #     features = tf.placeholder(tf.float32)
    #     label = tf.placeholder(tf.int32)
    #
    #     sess = tf.Session()
    #
    # while True:
    #     try:
    #         batch = [next(iterator) for _ in range(BATCH_SIZE)]
    #         _f = np.array([item[0] for item in batch])
    #         _l = np.array([item[1] for item in batch])
    #         f_data, l_data = sess.run([features, label], {features: _f, label: _l})
    #         print(f_data.shape, l_data)
    #     except StopIteration:
    #         break

    ################################
    '''
    METHOD 2: USING THE DATA API
    This is the preferred method by tensorflow as it makes use of the Data api 
    that they provide. We can define it as part of our graph and it is really 
    easy to run since we don't have to worry about feeding it into the graph 
    ourselves. This also gives us the higher level API access to do things like
    create batches, repeat the datasets, or other things. 
    
    NOTE: in this run, it will print the last batch even though it is not of the
    full batch size. The above can't do that because we don't pad the inputs and 
    the try-except breaks when we try to access the next element in from iterable 
    when it doesn't have any more left. The data API handles all this under the hood
    by itself. 
    '''
    dataset = pipeline.get_dataset()
    batched_ds = dataset.shuffle(buffer_size=2).batch(BATCH_SIZE).repeat(2)
    features, label = batched_ds.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
        while True:
            try:
                f_data, l_data = sess.run([features, label])
                print(f_data.shape, l_data)
            except tf.errors.OutOfRangeError:
                break
