import os
import math
import json
from socket import if_nameindex

from loguru import logger

import tensorflow as tf

from shuffle import shuffle


# TO PARSE:
@tf.function
def parse_item(item):
	parsed = tf.io.parse_single_example(item, features={
		"rainfallradar": tf.io.FixedLenFeature([], tf.string),
		"waterdepth": tf.io.FixedLenFeature([], tf.string)
	})
	rainfall = tf.io.parse_tensor(parsed["rainfallradar"], out_type=tf.float32)
	water = tf.io.parse_tensor(parsed["waterdepth"], out_type=tf.float32)
	
	# TODO: The shape of the resulting tensor can't be statically determined, so we need to reshape here
	
	# TODO: Any other additional parsing here, since multiple .map() calls are not optimal
	return rainfall, water

def make_dataset(filenames, compression_type="GZIP", parallel_reads_multiplier=1.5, shuffle_buffer_size=128, batch_size=64):
	return tf.data.TFRecordDataset(filenames,
		compression_type=compression_type,
		num_parallel_reads=math.ceil(os.cpu_count() * parallel_reads_multiplier)
	).shuffle(shuffle_buffer_size) \
		.map(parse_item, num_parallel_calls=tf.data.AUTOTUNE) \
		.batch(batch_size) \
		.prefetch(tf.data.AUTOTUNE)


def dataset(dirpath_input, batch_size=64, train_percentage=0.8, parallel_reads_multiplier=1.5):
	filepaths = shuffle(list(filter(
		lambda filepath: str(filepath).endswith(".tfrecord.gz"),
		[ file.path for file in os.scandir(dirpath_input) ] # .path on a DirEntry object yields the absolute filepath
	)))
	filepaths_count = len(filepaths)
	dataset_splitpoint = math.floor(filepaths_count * train_percentage)
	
	filepaths_train = filepaths[:dataset_splitpoint]
	filepaths_validate = filepaths[dataset_splitpoint:]
	
	dataset_train = make_dataset(filepaths_train, batch_size=batch_size, parallel_reads_multiplier=parallel_reads_multiplier)
	dataset_validate = make_dataset(filepaths_validate, batch_size=batch_size, parallel_reads_multiplier=parallel_reads_multiplier)
	
	return dataset_train, dataset_validate



if __name__ == "__main__":
	ds_train, ds_validate = dataset("/mnt/research-data/main/rainfallwater_records-viperfinal/")
	for thing in ds_validate():
		as_str = str(thing)
		print(thing[:200])