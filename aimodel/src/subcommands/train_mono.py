import math
import sys
import argparse

import tensorflow as tf

from lib.ai.RainfallWaterMono import RainfallWaterMono
from lib.dataset.dataset_mono import dataset_mono
from lib.dataset.read_metadata import read_metadata

def parse_args():
	parser = argparse.ArgumentParser(description="Train an mono rainfall-water model on a directory of .tfrecord.gz rainfall+waterdepth_label files.")
	# parser.add_argument("--config", "-c", help="Filepath to the TOML config file to load.", required=True)
	parser.add_argument("--input", "-i", help="Path to input directory containing the .tfrecord.gz files to pretrain with", required=True)
	parser.add_argument("--output", "-o", help="Path to output directory to write output to (will be automatically created if it doesn't exist)", required=True)
	parser.add_argument("--batch-size", help="Sets the batch size [default: 64].", type=int)
	parser.add_argument("--reads-multiplier", help="Optional. The multiplier for the number of files we should read from at once. Defaults to 1.5, which means read ceil(NUMBER_OF_CORES * 1.5) files at once. Set to a higher number of systems with high read latency to avoid starving the GPU of data.")
	parser.add_argument("--water-size", help="The width and height of the square of pixels that the model will predict. Smaller values crop the input more [default: 100].", type=int)
	parser.add_argument("--water-threshold", help="The threshold at which a water cell should be considered water. Water depth values lower than this will be set to 0 (no water). Value unit is metres [default: 0.1].", type=int)
	parser.add_argument("--bottleneck", help="The size of the bottleneck [default: 512].", type=int)
	parser.add_argument("--arch-enc", help="Next of the underlying encoder convnext model to use [default: convnext_xtiny].")
	parser.add_argument("--arch-dec", help="Next of the underlying decoder convnext model to use [default: convnext_i_xtiny].")
	
	
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
	if (not hasattr(args, "bottleneck")) or args.bottleneck == None:
		args.bottleneck = 512
	if (not hasattr(args, "arch_enc")) or args.arch_enc == None:
		args.arch_enc = "convnext_xtiny"
	if (not hasattr(args, "arch_dec")) or args.arch_dec == None:
		args.arch_dec = "convnext_i_xtiny"
	
	
	# TODO: Validate args here.
	
	sys.stderr.write(f"\n\n>>> This is TensorFlow {tf.__version__}\n\n\n")
	
	
	dataset_train, dataset_validate = dataset_mono(
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
	
	
	ai = RainfallWaterMono(
		dir_output=args.output,
		batch_size=args.batch_size,
		
		feature_dim=args.bottleneck,
		model_arch_enc=args.arch_enc,
		model_arch_dec=args.arch_dec,
		
		metadata = read_metadata(args.input),
		shape_water_out=[ args.water_size, args.water_size ], # The DESIRED output shape. the actual data will be cropped to match this.
	)
	
	ai.train(dataset_train, dataset_validate)
	