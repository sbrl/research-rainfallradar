import io
import json
import os
import sys
import argparse
import re

from loguru import logger
import tensorflow as tf
from lib.dataset.batched_iterator import batched_iterator

from lib.dataset.dataset_mono import dataset_mono_predict


MODE_JSONL = 1
MODE_PNG = 2

def parse_args():
	parser = argparse.ArgumentParser(description="Output water depth image segmentation maps using a given pretrained mono model.")
	# parser.add_argument("--config", "-c", help="Filepath to the TOML config file to load.", required=True)
	parser.add_argument("--input", "-i", help="Path to input directory containing the .tfrecord(.gz) files to predict for. If a single file is passed instead, then only that file will be converted.", required=True)
	parser.add_argument("--reads-multiplier", help="Optional. The multiplier for the number of files we should read from at once. Defaults to 0. When using this start with 1.5, which means read ceil(NUMBER_OF_CORES * 1.5). Set to a higher number of systems with high read latency to avoid starving the GPU of data. SETTING THIS WILL SCRAMBLE THE ORDER OF THE DATASET.", type=int)
	parser.add_argument("--batch-size", help="Optional. The batch size to calculate statistics with. Can be larger than normal since we don't have a model loaded. Default: 1024", type=int)
	return parser

def run(args):
	
	if (not hasattr(args, "read_multiplier")) or args.read_multiplier == None:
		args.read_multiplier = 4
	if (not hasattr(args, "batch_size")) or args.batch_size == None:
		args.batch_size = 1024
	
	sys.stderr.write(f"\n\n>>> This is TensorFlow {tf.__version__}\n\n\n")
	
	# Note that if using a directory of input files, the output order is NOT GUARANTEED TO BE THE SAME. In fact, it probably won't be (see dataset_mono for more details).
	dataset = dataset_mono_predict(
		dirpath_input=args.input,
		parallel_reads_multiplier=args.read_multiplier
	)
	
	# for items in dataset_train.repeat(10):
	# 	print("ITEMS", len(items))
	# 	print("LEFT", [ item.shape for item in items[0] ])
	# print("ITEMS DONE")
	# exit(0)
	
	
	logger.info("RAINFALL STATS")
	
	calc_mean = []
	calc_stddev = []
	calc_max = []
	
	i = 0
	for batch in batched_iterator(dataset, tensors_in_item=2, batch_size=args.batch_size):
		rainfall_actual_batch, water_actual_batch = batch
		
		rainfall_flat = tf.reshape(rainfall_actual_batch, [-1])
		
		batch_mean = tf.math.reduce_mean(rainfall_flat)
		batch_stddev = tf.math.reduce_std(rainfall_flat)
		batch_max = tf.math.reduce_max(rainfall_flat)
		
		print("BATCH mean\t", batch_mean.numpy().tolist(), "\tstddev\t", batch_stddev.numpy().tolist(), "\tmax\t", batch_max.numpy().tolist())
		
		calc_mean.append(batch_mean)
		calc_stddev.append(batch_stddev)
		calc_max.append(batch_max)
		
		i += 1
	
	
	calc_mean	= tf.math.reduce_mean(tf.stack(calc_mean))
	calc_max	= tf.math.reduce_max(tf.stack(calc_max))
	
	print("STDDEV VALUES", tf.stack(calc_stddev).numpy().tolist())
	print("OVERALL", "mean", calc_mean.numpy().tolist(), "max", calc_max.numpy().tolist())
	
	logger.write(">>> Complete\n")