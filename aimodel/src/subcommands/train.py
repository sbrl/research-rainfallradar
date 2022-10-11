import math
import sys
import argparse
from asyncio.log import logger

import tensorflow as tf

from lib.ai.RainfallWaterSegmenter import RainfallWaterSegmenter
from lib.dataset.dataset_segmenter import dataset_segmenter
from lib.dataset.read_metadata import read_metadata

def parse_args():
	parser = argparse.ArgumentParser(description="Train an image segmentation model on a directory of .tfrecord.gz embedded_rainfall+waterdepth_label files.")
	# parser.add_argument("--config", "-c", help="Filepath to the TOML config file to load.", required=True)
	parser.add_argument("--input", "-i", help="Path to input directory containing the .tfrecord.gz files to pretrain with", required=True)
	parser.add_argument("--output", "-o", help="Path to output directory to write output to (will be automatically created if it doesn't exist)", required=True)
	parser.add_argument("--feature-dim", help="The size of the input feature dimension of the model [default: 512].", type=int)
	parser.add_argument("--batch-size", help="Sets the batch size [default: 64].", type=int)
	parser.add_argument("--reads-multiplier", help="Optional. The multiplier for the number of files we should read from at once. Defaults to 1.5, which means read ceil(NUMBER_OF_CORES * 1.5) files at once. Set to a higher number of systems with high read latency to avoid starving the GPU of data.")
	parser.add_argument("--water-size", help="The width and height of the square of pixels that the model will predict. Smaller values crop the input more [default: 100].", type=int)
	parser.add_argument("--water-threshold", help="The threshold at which a water cell should be considered water. Water depth values lower than this will be set to 0 (no water). Value unit is metres [default: 0.1].", type=int)
	parser.add_argument("--arch", help="Next fo the underlying convnext model to use [default: 0.1].", type=int)
	
	
	return parser

def run(args):
	if (not hasattr(args, "water_size")) or args.water_size == None:
		args.water_size = 100
	if (not hasattr(args, "batch_size")) or args.batch_size == None:
		args.batch_size = 64
	if (not hasattr(args, "feature_dim")) or args.feature_dim == None:
		args.feature_dim = 512
	if (not hasattr(args, "read_multiplier")) or args.read_multiplier == None:
		args.read_multiplier = 1.5
	if (not hasattr(args, "water_threshold")) or args.water_threshold == None:
		args.water_threshold = 1.5
	if (not hasattr(args, "water_size")) or args.water_size == None:
		args.water_size = 1.5
	if (not hasattr(args, "arch")) or args.arch == None:
		args.arch = "convnext_i_xtiny"
	
	
	# TODO: Validate args here.
	
	sys.stderr.write(f"\n\n>>> This is TensorFlow {tf.__version__}\n\n\n")
	
	dataset_train, dataset_validate = dataset_segmenter(
		dirpath_input=args.input,
		batch_size=args.batch_size,
		water_threshold=args.water_threshold,
		shape_water_desired=[args.water_size, args.water_size]
	)
	dataset_metadata = read_metadata(args.input)
	
	# for (items, label) in dataset_train:
	# 	print("ITEMS", len(items), [ item.shape for item in items ])
	# 	print("LABEL", label.shape)
	# print("ITEMS DONE")
	# exit(0)
	
	
	ai = RainfallWaterSegmenter(
		dir_output=args.output,
		batch_size=args.batch_size,
		feature_dim_in=args.feature_dim,
		
		model_arch=args.arch,
		metadata = read_metadata(args.input),
		shape_water_out=[ args.water_size, args.water_size ], # The DESIRED output shape. the actual data will be cropped to match this.
	)
	
	ai.train(dataset_train, dataset_validate)
	