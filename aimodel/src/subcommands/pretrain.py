import math
import sys
import argparse
from asyncio.log import logger

import tensorflow as tf

from lib.ai.RainfallWaterContraster import RainfallWaterContraster
from lib.dataset.dataset import dataset
from lib.dataset.read_metadata import read_metadata

def parse_args():
	parser = argparse.ArgumentParser(description="Pretrain a contrastive learning model on a directory of rainfall+water .tfrecord.gz files.")
	# parser.add_argument("--config", "-c", help="Filepath to the TOML config file to load.", required=True)
	parser.add_argument("--input", "-i", help="Path to input directory containing the .tfrecord.gz files to pretrain with", required=True)
	parser.add_argument("--output", "-o", help="Path to output directory to write output to (will be automatically created if it doesn't exist)", required=True)
	parser.add_argument("--feature-dim", help="The size of the output feature dimension of the model [default: 2048].", type=int)
	parser.add_argument("--batch-size", help="Sets the batch size [default: 64].", type=int)
	parser.add_argument("--reads-multiplier", help="Optional. The multiplier for the number of files we should read from at once. Defaults to 1.5, which means read ceil(NUMBER_OF_CORES * 1.5) files at once. Set to a higher number of systems with high read latency to avoid starving the GPU of data.")
	parser.add_argument("--water-size", help="The width and height of the square of pixels that the model will predict. Smaller values crop the input more [default: 100].", type=int)
	
	return parser


def count_batches(dataset):
	count = 0
	for _ in dataset:
		count += 1
	return count

def run(args):
	if (not hasattr(args, "water_size")) or args.water_size == None:
		args.water_size = 100
	if (not hasattr(args, "batch_size")) or args.batch_size == None:
		args.batch_size = 64
	if (not hasattr(args, "feature_dim")) or args.feature_dim == None:
		args.feature_dim = 2048
	if (not hasattr(args, "read_multiplier")) or args.read_multiplier == None:
		args.read_multiplier = 1.5
	
	
	# TODO: Validate args here.
	
	sys.stderr.write(f"\n\n>>> This is TensorFlow {tf.__version__}\n\n\n")
	
	dataset_train, dataset_validate = dataset(
		dirpath_input=args.input,
		batch_size=args.batch_size,
	)
	
	print("BATCHES_TRAIN", count_batches(dataset_train))
	print("BATCHES_VALIDATE", count_batches(dataset_validate))
	
	
	
	# for (items, label) in dataset_train:
	# 	print("ITEMS", len(items), [ item.shape for item in items ])
	# 	print("LABEL", label.shape)
	# print("ITEMS DONE")
	# exit(0)
	
	
	ai = RainfallWaterContraster(
		dir_output=args.output,
		batch_size=args.batch_size,
		feature_dim=args.feature_dim,
		
		metadata = read_metadata(args.input),
		shape_water=[ args.water_size, args.water_size ] # The DESIRED output shape. the actual data will be cropped to match this.
	)
	
	ai.train(dataset_train, dataset_validate)
