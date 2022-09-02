import os
import math
import json

from loguru import logger

import tensorflow as tf

from lib.dataset.read_metadata import read_metadata

from ..io.readfile import readfile
from .shuffle import shuffle



# TO PARSE:
def parse_item(metadata, shape_water_desired):
	water_width_source, water_height_source = metadata["waterdepth"]
	water_width_target, water_height_target = shape_water_desired
	water_offset_x = math.ceil((water_width_source - water_width_target) / 2)
	water_offset_y = math.ceil((water_height_source - water_height_target) / 2)
	def parse_item_inner(item):
		parsed = tf.io.parse_single_example(item, features={
			"rainfallradar": tf.io.FixedLenFeature([], tf.string),
			"waterdepth": tf.io.FixedLenFeature([], tf.string)
		})
		rainfall = tf.io.parse_tensor(parsed["rainfallradar"], out_type=tf.float32)
		water = tf.io.parse_tensor(parsed["waterdepth"], out_type=tf.float32)
		# [channels, width, height] → [width, height, channels] - ref ConvNeXt does not support data_format=channels_first
		
		rainfall = tf.reshape(rainfall, tf.constant(metadata["rainfallradar"], dtype=tf.int32))
		water = tf.reshape(water, tf.constant(metadata["waterdepth"], dtype=tf.int32))
		
		rainfall = tf.transpose(rainfall, [1, 2, 0])
		rainfall = tf.image.resize(rainfall, tf.cast(tf.constant(metadata["waterdepth"]) / 2, dtype=tf.int32))
		
		water = tf.image.crop_to_bounding_box(water, water_offset_x, water_offset_y, water_width_target, water_height_target)
		water = tf.expand_dims(water, axis=-1) # [width, height] → [width, height, channels]
		
		# TODO: The shape of the resulting tensor can't be statically determined, so we need to reshape here
		print("DEBUG:dataset ITEM rainfall:shape", rainfall.shape, "water:shape", water.shape)
		# TODO: Any other additional parsing here, since multiple .map() calls are not optimal
		return ((rainfall, water), tf.ones(1))
	
	return tf.function(parse_item_inner)

def make_dataset(filenames, metadata, shape_watch_desired=[100,100], compression_type="GZIP", parallel_reads_multiplier=1.5, shuffle_buffer_size=128, batch_size=64):
	if "NO_PREFETCH" in os.environ:
		logger.info("disabling data prefetching.")
	return tf.data.TFRecordDataset(filenames,
		compression_type=compression_type,
		num_parallel_reads=math.ceil(os.cpu_count() * parallel_reads_multiplier)
	).shuffle(shuffle_buffer_size) \
		.map(parse_item(metadata, shape_water_desired=shape_watch_desired), num_parallel_calls=tf.data.AUTOTUNE) \
		.batch(batch_size) \
		.prefetch(0 if "NO_PREFETCH" in os.environ else tf.data.AUTOTUNE)


def dataset(dirpath_input, batch_size=64, train_percentage=0.8, parallel_reads_multiplier=1.5):
	filepaths = shuffle(list(filter(
		lambda filepath: str(filepath).endswith(".tfrecord.gz"),
		[ file.path for file in os.scandir(dirpath_input) ] # .path on a DirEntry object yields the absolute filepath
	)))
	filepaths_count = len(filepaths)
	dataset_splitpoint = math.floor(filepaths_count * train_percentage)
	
	filepaths_train = filepaths[:dataset_splitpoint]
	filepaths_validate = filepaths[dataset_splitpoint:]
	
	metadata = read_metadata(dirpath_input)
	
	dataset_train = make_dataset(filepaths_train, metadata, batch_size=batch_size, parallel_reads_multiplier=parallel_reads_multiplier)
	dataset_validate = make_dataset(filepaths_validate, metadata, batch_size=batch_size, parallel_reads_multiplier=parallel_reads_multiplier)
	
	return dataset_train, dataset_validate #, filepaths

def dataset_predict():
	raise NotImplementedError("Not implemented yet")


if __name__ == "__main__":
	ds_train, ds_validate = dataset("/mnt/research-data/main/rainfallwater_records-viperfinal/")
	for thing in ds_validate():
		as_str = str(thing)
		print(thing[:200])