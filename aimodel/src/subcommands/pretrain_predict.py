import io
import json
import os
import sys
import argparse
import re

from loguru import logger
import tensorflow as tf
import numpy as np

from lib.io.writefile import writefile
from lib.io.handle_open import handle_open
from lib.ai.RainfallWaterContraster import RainfallWaterContraster
from lib.dataset.dataset import dataset_predict
from lib.io.find_paramsjson import find_paramsjson
from lib.io.readfile import readfile
from lib.vis.embeddings import vis_embeddings


MODE_JSONL = 1
MODE_TFRECORD = 2

def parse_args():
	parser = argparse.ArgumentParser(description="Output feature maps using a given pretrained contrastive model.")
	# parser.add_argument("--config", "-c", help="Filepath to the TOML config file to load.", required=True)
	parser.add_argument("--input", "-i", help="Path to input directory containing the .tfrecord(.gz) files to predict for. If a single file is passed instead, then only that file will be converted.", required=True)
	parser.add_argument("--output", "-o", help="Path to output file to write output to. If the file extension .tfrecord.gz is used instead of .jsonl.gz, then a tfrecord file is written.")
	parser.add_argument("--records-per-file", help="Optional. If specified, this limits the number of records written to each file. When using this option, you MUST have the string '+d' (without quotes) somewhere in your output filepath.", type=int)
	parser.add_argument("--checkpoint", "-c", help="Checkpoint file to load model weights from.", required=True)
	parser.add_argument("--params", "-p", help="Optional. The file containing the model hyperparameters (usually called 'params.json'). If not specified, it's location will be determined automatically.")
	parser.add_argument("--reads-multiplier", help="Optional. The multiplier for the number of files we should read from at once. Defaults to 1.5, which means read ceil(NUMBER_OF_CORES * 1.5). Set to a higher number of systems with high read latency to avoid starving the GPU of data.")
	
	return parser

def run(args):
	
	# Note that we do NOT check to see if the checkpoint file exists, because Tensorflow/Keras requires that we pass the stem instead of the actual index file..... :-/
	
	if (not hasattr(args, "params")) or args.params == None:
		args.params = find_paramsjson(args.checkpoint)
	if (not hasattr(args, "read_multiplier")) or args.read_multiplier == None:
		args.read_multiplier = 1.5
	if (not hasattr(args, "records_per_file")) or args.records_per_file == None:
		args.records_per_file = 0 # 0 = unlimited
	
	if not os.path.exists(args.params):
		raise Exception(f"Error: The specified filepath params.json hyperparameters ('{args.params}) does not exist.")
	if not os.path.exists(args.checkpoint):
		raise Exception(f"Error: The specified filepath to the checkpoint to load ('{args.checkpoint}) does not exist.")
	
	
	filepath_output = args.output if hasattr(args, "output") and args.output != None else "-"
	
	
	ai = RainfallWaterContraster.from_checkpoint(args.checkpoint, **json.loads(readfile(args.params)))
	
	sys.stderr.write(f"\n\n>>> This is TensorFlow {tf.__version__}\n\n\n")
	
	# Note that if using a directory of input files, the output order is NOT GUARANTEED TO BE THE SAME. In fact, it probably won't be.
	dataset = dataset_predict(
		dirpath_input=args.input,
		parallel_reads_multiplier=args.read_multiplier
	)
	
	# for items in dataset_train.repeat(10):
	# 	print("ITEMS", len(items))
	# 	print("LEFT", [ item.shape for item in items[0] ])
	# print("ITEMS DONE")
	# exit(0)
	
	output_mode = MODE_TFRECORD if filepath_output.endswith(".tfrecord") or filepath_output.endswith(".tfrecord.gz") else MODE_JSONL
	
	logger.info("Output mode is "+("TFRECORD" if output_mode == MODE_TFRECORD else "JSONL"))
	logger.info(f"Records per file: {args.records_per_file}")
	
	write_mode = "wt" if filepath_output.endswith(".gz") else "w"
	if output_mode == MODE_TFRECORD:
		write_mode = "wb"
	
	handle = sys.stdout
	filepath_params = None
	if filepath_output != "-":
		handle = handle_open(
			filepath_output if args.records_per_file <= 0 else filepath_output.replace("+d", str(0)),
			write_mode
		)
		filepath_params = os.path.join(os.path.dirname(filepath_output), "params.json")
	
	logger.info(f"filepath_output: {filepath_output}")
	logger.info(f"filepath_params: {filepath_params}")
	
	i = 0
	i_file = i
	files_done = 0
	for step_rainfall, step_water in ai.embed(dataset):
		if args.records_per_file > 0 and i_file > args.records_per_file:
			files_done += 1
			i_file = 0
			handle.close()
			logger.info(f"PROGRESS:file {files_done}")
			handle = handle_open(filepath_output.replace("+d", str(files_done+1)), write_mode)
		
		if output_mode == MODE_JSONL:
			handle.write(json.dumps(step_rainfall.numpy().tolist(), separators=(',', ':'))+"\n") # Ref https://stackoverflow.com/a/64710892/1460422
		elif output_mode == MODE_TFRECORD:
			if i == 0 and filepath_params is not None:
				writefile(filepath_params, json.dumps({
					"rainfallradar": step_rainfall.shape.as_list(),
					"waterdepth": step_water.shape.as_list()
				}))
			step_rainfall = tf.train.BytesList(value=[tf.io.serialize_tensor(step_rainfall, name="rainfall").numpy()])
			step_water = tf.train.BytesList(value=[tf.io.serialize_tensor(step_water, name="water").numpy()])
			
			record = tf.train.Example(features=tf.train.Features(feature={
				"rainfallradar": tf.train.Feature(bytes_list=step_rainfall),
				"waterdepth": tf.train.Feature(bytes_list=step_water)
			}))
			handle.write(record.SerializeToString())
		else:
			raise Exception("Error: Unknown output mode.")
		
		if i == 0 or i % 100 == 0:
			sys.stderr.write(f"[pretrain:predict] STEP {i}\r")
		
		i += 1
		i_file += 1
	
	handle.close()
	
	sys.stderr.write(">>> Complete\n")