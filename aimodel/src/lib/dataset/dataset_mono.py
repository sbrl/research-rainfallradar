import os
import math
import json

from loguru import logger

import tensorflow as tf

from lib.dataset.read_metadata import read_metadata

from ..io.readfile import readfile
from .parse_heightmap import parse_heightmap
from .primitives.shuffle import shuffle
from .primitives.remove_isolated_pixels import remove_isolated_pixels


# TO PARSE:
def parse_item(metadata, output_size=100, input_size="same", water_threshold=0.1, water_bins=2, heightmap=None, rainfall_scale_up=1, do_remove_isolated_pixels=True):
	if input_size == "same":
		input_size = output_size # This is almost always the case with e.g. the DeepLabV3+ model
	
	water_height_source, water_width_source = metadata["waterdepth"]
	water_offset_x = math.ceil((water_width_source - output_size) / 2)
	water_offset_y = math.ceil((water_height_source - output_size) / 2)
	
	_, rainfall_height_source, rainfall_width_source = metadata["rainfallradar"]
	rainfall_height_source *= rainfall_scale_up
	rainfall_width_source *= rainfall_scale_up
	rainfall_offset_x = math.ceil((rainfall_width_source - input_size) / 2)
	rainfall_offset_y = math.ceil((rainfall_height_source - input_size) / 2)
	
	print("DEBUG DATASET:rainfall shape", metadata["rainfallradar"], "/", f"w {rainfall_width_source} h {rainfall_height_source}")
	print("DEBUG DATASET:water shape", metadata["waterdepth"])
	print("DEBUG DATASET:water_threshold", water_threshold)
	print("DEBUG DATASET:water_bins", water_bins)
	print("DEBUG DATASET:output_size", output_size)
	print("DEBUG DATASET:input_size", input_size)
	print("DEBUG DATASET:water_offset x", water_offset_x, "y", water_offset_y)
	print("DEBUG DATASET:rainfall_offset x", rainfall_offset_x, "y", rainfall_offset_y)
	
	if heightmap is not None:
		heightmap = tf.expand_dims(heightmap, axis=-1)
		norm = tf.keras.layers.Normalization(axis=None)
		norm.adapt(heightmap)
		# THIS IS (probably) OK, because BatchNorm also outputs mean=0 stddev=1, bias term shifts anyway
		# Ref https://datascience.stackexchange.com/a/54383/86851
		heightmap = norm(heightmap)
		heightmap = tf.transpose(heightmap, [1, 0, 2]) # [width, height] → [height, width]
	
	def parse_item_inner(item):
		parsed = tf.io.parse_single_example(item, features={
			"rainfallradar": tf.io.FixedLenFeature([], tf.string),
			"waterdepth": tf.io.FixedLenFeature([], tf.string)
		})
		rainfall = tf.io.parse_tensor(parsed["rainfallradar"], out_type=tf.float32)
		water = tf.io.parse_tensor(parsed["waterdepth"], out_type=tf.float32)
		
		
		rainfall = tf.reshape(rainfall, tf.constant(metadata["rainfallradar"], dtype=tf.int32))
		water = tf.reshape(water, tf.constant(metadata["waterdepth"], dtype=tf.int32))
		
		# Apparently the water depth data is also in HW instead of WH.... sighs
		# * YES IT IS, BUT TENSORFLOW *wants* NHWC NOT NWHC....!
		# water = tf.transpose(water, [1, 0])
		
		# [channels, height, weight] → [height, width, channels] - ref ConvNeXt does not support data_format=channels_first
		# BUG: For some reasons we have data that's not transposed correctly still!! O.o
		# I can't believe in this entire project I have yet to get the rotation of the rainfall radar data correct....!
		# %TRANSPOSE%
		rainfall = tf.transpose(rainfall, [1, 2, 0])
		if heightmap is not None:
			rainfall = tf.concat([rainfall, heightmap], axis=-1)
		if rainfall_scale_up > 1:
			rainfall = tf.repeat(tf.repeat(rainfall, rainfall_scale_up, axis=0), rainfall_scale_up, axis=1)
		if input_size is not None:
			rainfall = tf.image.crop_to_bounding_box(rainfall,
				offset_width=rainfall_offset_x,
				offset_height=rainfall_offset_y,
				target_width=input_size,
				target_height=input_size,
			)
		
		# rainfall = tf.image.resize(rainfall, tf.cast(tf.constant(metadata["rainfallradar"]) / 2, dtype=tf.int32))
		water = tf.expand_dims(water, axis=-1) # [height, width] → [height, width, channels=1]
		water = tf.image.crop_to_bounding_box(water,
			offset_width=water_offset_x,
			offset_height=water_offset_y,
			target_width=output_size,
			target_height=output_size
		)
		
		print("DEBUG:dataset BEFORE_SQUEEZE water", water.shape)
		water = tf.squeeze(water)
		print("DEBUG:dataset AFTER_SQUEEZE water", water.shape)
		# ONE-HOT [LOSS cross entropy]
		# water = tf.cast(tf.math.greater_equal(water, water_threshold), dtype=tf.int32)
		# water = tf.one_hot(water, water_bins, axis=-1, dtype=tf.int32)
		# SPARSE [LOSS dice / sparse cross entropy]
		water = tf.cast(tf.math.greater_equal(water, water_threshold), dtype=tf.float32)
		if do_remove_isolated_pixels:
			water = remove_isolated_pixels(water)
		
		print("DEBUG DATASET_OUT:rainfall shape", rainfall.shape)
		print("DEBUG DATASET_OUT:water shape", water.shape)
		return rainfall, water
	
	return tf.function(parse_item_inner)

def make_dataset(filepaths, compression_type="GZIP", parallel_reads_multiplier=1.5, shuffle_buffer_size=128, batch_size=64, prefetch=True, shuffle=True, filepath_heightmap=None, **kwargs):
	if "NO_PREFETCH" in os.environ:
		logger.info("disabling data prefetching.")
	
	heightmap = None
	if filepath_heightmap is not None:
		logger.info(f"Using heightmap from '{filepath_heightmap}'.")
		heightmap = parse_heightmap(filepath_heightmap)
	
	dataset = tf.data.TFRecordDataset(filepaths,
		compression_type=compression_type,
		num_parallel_reads=math.ceil(os.cpu_count() * parallel_reads_multiplier) if parallel_reads_multiplier > 0 else None
	)
	if shuffle:
		dataset = dataset.shuffle(shuffle_buffer_size)
	dataset = dataset.map(parse_item(heightmap=heightmap, **kwargs), num_parallel_calls=tf.data.AUTOTUNE)
	
	if batch_size != None:
		dataset = dataset.batch(batch_size, drop_remainder=True)
	if prefetch:
		dataset = dataset.prefetch(0 if "NO_PREFETCH" in os.environ else tf.data.AUTOTUNE)
	
	return dataset


def get_filepaths(dirpath_input, do_shuffle=True):
	result = list(filter(
		lambda filepath: str(filepath).endswith(".tfrecord.gz"),
		[ file.path for file in os.scandir(dirpath_input) ] # .path on a DirEntry object yields the absolute filepath
	))
	if do_shuffle:
		result = shuffle(result)
	else:
		result = sorted(result, key=lambda filepath: int(os.path.basename(filepath).split(".", 1)[0]))
	
	return result

def dataset_mono(dirpath_input, train_percentage=0.8, **kwargs):
	filepaths = get_filepaths(dirpath_input)
	filepaths_count = len(filepaths)
	dataset_splitpoint = math.floor(filepaths_count * train_percentage)
	
	filepaths_train = filepaths[:dataset_splitpoint]
	filepaths_validate = filepaths[dataset_splitpoint:]
	
	metadata = read_metadata(dirpath_input)
	
	dataset_train = make_dataset(filepaths_train, metadata=metadata, **kwargs)
	dataset_validate = make_dataset(filepaths_validate, metadata=metadata, **kwargs)
	
	return dataset_train, dataset_validate #, filepaths

def dataset_mono_predict(dirpath_input, batch_size=64, **kwargs):
	"""Creates a tf.data.Dataset() for prediction using the contrastive learning model.
	Note that this WILL MANGLE THE ORDERING if you set parallel_reads_multiplier to anything other than 0!!
	
	Args:
		dirpath_input (string): The path to the directory containing the input (.tfrecord.gz) files
		parallel_reads_multiplier (float, optional): The number of files to read in parallel. Defaults to 1.5.
		prefetch (bool, optional): Whether to prefetch data into memory or not. Defaults to True.

	Returns:
		tf.data.Dataset: A tensorflow Dataset for the given input files.
	"""
	filepaths = get_filepaths(dirpath_input, do_shuffle=False) if os.path.isdir(dirpath_input) else [ dirpath_input ]
	
	return make_dataset(
		filepaths=filepaths,
		metadata=read_metadata(dirpath_input),
		batch_size=batch_size, # WAS None
		shuffle=False, #even with shuffle=False we're not gonna get them all in the same order since we're reading in parallel
		**kwargs
	)

if __name__ == "__main__":
	ds_train, ds_validate = dataset_mono("/mnt/research-data/main/rainfallwater_records-viperfinal/")
	for thing in ds_validate():
		as_str = str(thing)
		print(thing[:200])