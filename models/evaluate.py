#!/usr/bin/env python
from __future__ import division

import tensorflow as tf
import argparse

import i3d
from pipeline import *

MAX_ITER = 10

def inference(rgb_inputs):



def evaluate(input_file, ckpt_dir, top_k=None):
    with tf.Graph().as_default() as g:
        eval_pipeline = Pipeline(input_file)
        eval_queue = eval_pipeline.get_dataset().shuffle(buffer_size=3).batch(2).repeat(MAX_ITER)
        rgbs,labels = eval_queue.make_one_shot_iterator().get_next()
        
        rgbs, label = pipeline
        rgbs, flows, labels = pipeline.get_batch(train=False)
        rgb_logits, flow_logits = inference(rgbs, flows)
        model_logits = rgb_logits + flow_logits
        top_k_op = tf.nn.in_top_k(model_logits, labels, 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('--ckpt_dir', required=True)
    parser.add_argument('--top_k')
    args = parser.parse_args()
    evaluate(args.input_file, args.ckpt_dir, args.top_k)
