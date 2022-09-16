import json
import os
import sys
import argparse

from loguru import logger
import tensorflow as tf
import numpy as np

from lib.io.handle_open import handle_open
from lib.vis.embeddings import vis_embeddings

def parse_args():
	parser = argparse.ArgumentParser(description="Plot embeddings predicted by the contrastive learning pretrained model with UMAP.")
	# parser.add_argument("--config", "-c", help="Filepath to the TOML config file to load.", required=True)
	parser.add_argument("--input", "-i", help="Path to input file containing the content to plot.", required=True)
	parser.add_argument("--output", "-o", help="Path to output file to write the resulting image to.", required=True)
	
	return parser

def run(args):
	
	# Note that we do NOT check to see if the checkpoint file exists, because Tensorflow/Keras requires that we pass the stem instead of the actual index file..... :-/
	
	
	if not os.path.exists(args.input):
		raise Exception(f"Error: The specified input filepath ('{args.input}) does not exist.")
	
	filepath_input = args.input
	
	stem, ext = os.path.splitext(args.output)
	filepath_output_rainfall = f"{stem}-rainfall.{ext}"
	filepath_output_water = f"{stem}-water.{ext}"
	
	
	sys.stderr.write(f"\n\n>>> This is TensorFlow {tf.__version__}\n\n\n")
	
	embeddings = []
	with handle_open(filepath_input, "r") as handle:
		for line in handle:
			obj = json.loads(line)
			embeddings.append(obj)
	
	logger.info(">>> Plotting rainfall with UMAP\n")
	vis_embeddings(filepath_output_rainfall, np.array(embeddings))
	
	# the model doesn't save the water encoder at this time
	# embeddings = []
	# with handle_open(filepath_input, "r") as handle:
	# 	for line in handle:
	# 		obj = json.loads(line)
	# 		embeddings.append(obj["water"])
	
	# logger.info(">>> Plotting water with UMAP\n")
	# vis_embeddings(filepath_output_water, np.array(embeddings))
	
	sys.stderr.write(">>> Complete\n")