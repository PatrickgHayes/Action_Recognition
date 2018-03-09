#!/usr/bin/env python
from __future__ import division

import tensorflow as tf
import i3d
from pipeline import Pipeline
from config import *
from config import VAL_DATA
import argparse
import numpy as np

# build the model
def inference(rgb_inputs):
    with tf.variable_scope('RGB'):
        rgb_model = i3d.InceptionI3d(
                NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
        rgb_logits, _ = rgb_model(rgb_inputs, is_training=True, dropout_keep_prob=1.0)
    return rgb_logits

def evaluate(input_file=VAL_DATA, ckpt_dir="./tmp/ckpt", top_k=None):
    with tf.Graph().as_default() as g:
        pipeline = Pipeline(input_file, CLS_DICT_FP)
        queue = pipeline.get_dataset().shuffle(buffer_size=10).batch(5)
        iterator = tf.contrib.data.Iterator.from_structure(queue.output_types, queue.output_shapes)
        init_op = iterator.make_initializer(queue)
        rgbs, labels = iterator.get_next()
        rgb_logits = inference(rgbs)
        model_logits = rgb_logits
        top_k_op = tf.nn.in_top_k(model_logits, labels, 1)

        if top_k:
            prob_op = tf.nn.softmax(model_logits)
            cls_dict = {}
            with open(REVERSE_CLS_DICT_FP, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    ind, cls_name = line.split(' ')
                    cls_dict[int(ind) - 1] = cls_name.lower()

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            ckpt = tf.train.get_checkpoint_state(ckpt_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print ('Restoring from:', ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print ('error restoring ckpt...')
                return

            sess.run(init_op)
            if top_k:
                with open('out_prob.txt', 'w+') as f:
                    while True:
                        try:
                            probs, cls_labels = sess.run([prob_op, labels])
                            indices = np.argsort(probs)
                            for i in range(indices.shape[0]):
                                f.write('true class: ' + cls_dict[cls_labels[i]] + '\n')
                                for j in range(indices.shape[1]):
                                    f.write(cls_dict[indices[i, j]] + '\t' + probs[i, j] + '\n')
                                vIdx += 1
                                f.write('\n\n')
                        except tf.errors.OutOfRangeError as e:
                            break

            else:
                true_count = 0
                while True:
                    try:
                        true_count += np.sum(sess.run(top_k_op))
                    except tf.errors.OutOfRangeError as e:
                        break
                print('eval accuracy: %.3f' % (true_count / len(pipeline.videos)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file')
    parser.add_argument('--ckpt_dir')
    parser.add_argument('--top_k')
    args = parser.parse_args()
    #evaluate(args.input_file, args.ckpt_dir, args.top_k)
    evaluate()

