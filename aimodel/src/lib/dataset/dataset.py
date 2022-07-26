import os
import math
import json
from socket import if_nameindex

from loguru import logger

import tensorflow as tf

from shuffle import shuffle

def parse_line(line):
	if tf.strings.length(line) <= 0:
		return None
	try:
		# Yes, this is really what the function is called that converts a string tensor to a regular python string.....
		obj = json.loads(line.numpy())
	except:
		logger.warn("Ignoring invalid line.")
		return None
	
	rainfall = tf.constant(obj.rainfallradar, dtype=tf.float32)
	waterdepth = tf.constant(obj.waterdepth, dtype=tf.float32)
	
	# Inputs, dummy label since we'll be using semi-supervised contrastive learning
	return rainfall, waterdepth

def make_dataset(filepaths, batch_size, shuffle_buffer_size=128, parallel_reads_multiplier=2):
	return tf.data.TextLineDataset(
		filenames=tf.data.Dataset.from_tensor_slices(filepaths).shuffle(len(filepaths), reshuffle_each_iteration=True),
		compression_type=tf.constant("GZIP"),
		num_parallel_reads=math.ceil(os.cpu_count() * parallel_reads_multiplier) # iowait can cause issues - especially on Viper
		# TODO: Get rid of this tf.py_function call somehow, because it acquires the Python Global Interpreter lock, which prevents more than 1 thread to run at a time, and .map() uses threads....
	).map(tf.py_function(parse_line), num_parallel_calls=tf.data.AUTOTUNE) \
		.filter(lambda item : item is not None) \
		.shuffle(1) \
		.batch(batch_size) \
		.prefetch(tf.data.AUTOTUNE)
	

def dataset(dirpath_input, batch_size=64, train_percentage=0.8):
	filepaths = shuffle(list(filter(
		lambda filepath: str(filepath).endswith(".jsonl.gz"),
		[ file.path for file in os.scandir(dirpath_input) ] # .path on a DirEntry object yields the absolute filepath
	)))
	filepaths_count = len(filepaths)
	dataset_splitpoint = math.floor(filepaths_count * train_percentage)
	
	filepaths_train = filepaths[:dataset_splitpoint]
	filepaths_validate = filepaths[dataset_splitpoint:]
	
	dataset_train = make_dataset(filepaths_train, batch_size)
	dataset_validate = make_dataset(filepaths_validate, batch_size)
	return dataset_train, dataset_validate



if __name__ == "__main__":
	ds_train, ds_validate = dataset("/mnt/research-data/main/rainfallwater_records-viperfinal/")
	for thing in ds_validate():
		as_str = str(thing)
		print(thing[:200])