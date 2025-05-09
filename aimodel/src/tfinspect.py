#!/usr/bin/env python3
from datetime import datetime

from loguru import logger
import tensorflow as tf

import lib.primitives.env as env


time_start = datetime.now()
logger.info(f"Starting at {str(datetime.now().isoformat())}")
logger.info(f"I, Tensorflow am version {tf.__version__}")


# ███████ ███    ██ ██    ██ ██ ██████   ██████  ███    ██ ███    ███ ███████ ███    ██ ████████
# ██      ████   ██ ██    ██ ██ ██   ██ ██    ██ ████   ██ ████  ████ ██      ████   ██    ██
# █████   ██ ██  ██ ██    ██ ██ ██████  ██    ██ ██ ██  ██ ██ ████ ██ █████   ██ ██  ██    ██
# ██      ██  ██ ██  ██  ██  ██ ██   ██ ██    ██ ██  ██ ██ ██  ██  ██ ██      ██  ██ ██    ██
# ███████ ██   ████   ████   ██ ██   ██  ██████  ██   ████ ██      ██ ███████ ██   ████    ██

FILEPATH_TFRECORD = env.read("FILEPATH_TFRECORD", str)



env.val_file_exists(FILEPATH_TFRECORD)



logger.info("TFRecord File Inspector")

env.print_all()


# ----------------------

@tf.function
def preprocess(sample):
	parsed = tf.io.parse_single_example(
		sample,
		features={
			"rainfallradar": tf.io.FixedLenFeature([], tf.string),
			"waterdepth": tf.io.FixedLenFeature([], tf.string),
		},
	)
	rainfall = tf.io.parse_tensor(parsed["rainfallradar"], out_type=tf.float32)
	water = tf.io.parse_tensor(parsed["waterdepth"], out_type=tf.float32)
	
	return rainfall, water


# ----------------------


compression_type="GZIP" if FILEPATH_TFRECORD.endswith(".gz") else ""

dataset = tf.data.TFRecordDataset([FILEPATH_TFRECORD], compression_type=compression_type)
dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)


for i, batch in enumerate(dataset):
	print(f"SAMPLE#{str(i)}", batch)
	
	print("> ITEM#1 UNIQUES", tf.unique(tf.reshape(batch[1], [-1])))
	
	if i > 3:
		break