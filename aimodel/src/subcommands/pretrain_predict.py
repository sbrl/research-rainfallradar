import io
import json
import os
import sys
import argparse
import re

from loguru import logger
import tensorflow as tf
import numpy as np

from lib.ai.RainfallWaterContraster import RainfallWaterContraster
from lib.dataset.dataset import dataset_predict
from lib.io.find_paramsjson import find_paramsjson
from lib.io.readfile import readfile
from lib.vis.embeddings import vis_embeddings

def parse_args():
	parser = argparse.ArgumentParser(description="Output feature maps using a given pretrained contrastive model.")
	# parser.add_argument("--config", "-c", help="Filepath to the TOML config file to load.", required=True)
	parser.add_argument("--input", "-i", help="Path to input directory containing the images to predict for.", required=True)
	parser.add_argument("--output", "-o", help="Path to output file to write output to. Defaults to stdout, but if specified a UMAP graph will NOT be produced.")
	parser.add_argument("--checkpoint", "-c", help="Checkpoint file to load model weights from.", required=True)
	parser.add_argument("--params", "-p", help="Optional. The file containing the model hyperparameters (usually called 'params.json'). If not specified, it's location will be determined automatically.")
	parser.add_argument("--reads-multiplier", help="Optional. The multiplier for the number of files we should read from at once. Defaults to 1.5, which means read ceil(NUMBER_OF_CORES * 1.5). Set to a higher number of systems with high read latency to avoid starving the GPU of data.")
	parser.add_argument("--no-vis",
		help="Don't also plot a visualisation of the resulting embeddings.", action="store_true")
	parser.add_argument("--only-gpu",
		help="If the GPU is not available, exit with an error (useful on shared HPC systems to avoid running out of memory & affecting other users)", action="store_true")
	
	return parser

def run(args):
	
	# Note that we do NOT check to see if the checkpoint file exists, because Tensorflow/Keras requires that we pass the stem instead of the actual index file..... :-/
	
	if (not hasattr(args, "params")) or args.params == None:
		args.params = find_paramsjson(args.checkpoint)
	if (not hasattr(args, "read_multiplier")) or args.read_multiplier == None:
		args.read_multiplier = 1.5
	
	if not os.path.exists(args.params):
		raise Exception(f"Error: The specified filepath params.json hyperparameters ('{args.params}) does not exist.")
	if not os.path.exists(args.checkpoint):
		raise Exception(f"Error: The specified filepath to the checkpoint to load ('{args.checkpoint}) does not exist.")
	
	
	filepath_output = args.output if hasattr(args, "output") and args.output != None else "-"
	
	
	ai = RainfallWaterContraster.from_checkpoint(args.checkpoint)
	
	sys.stderr.write(f"\n\n>>> This is TensorFlow {tf.__version__}\n\n\n")
	
	dataset = dataset_predict(
		dirpath_input=args.input,
		batch_size=ai.batch_size,
		parallel_reads_multiplier=args.read_multiplier
	)
	
	# for items in dataset_train.repeat(10):
	# 	print("ITEMS", len(items))
	# 	print("LEFT", [ item.shape for item in items[0] ])
	# print("ITEMS DONE")
	# exit(0)
	
	handle = sys.stdout
	if filepath_output != "-":
		handle = io.open(filepath_output, "w")
	
	for rainfall, water in ai.embed(dataset):
		handle.write(json.dumps({
			"rainfall": rainfall.numpy().tolist(),
			"water": water.numpy().tolist()
		}, separators=(',', ':'))+"\n") # Ref https://stackoverflow.com/a/64710892/1460422
	
	handle.close()
	
	sys.stderr.write(">>> Complete\n")