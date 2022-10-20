import io
import json
import os
import sys
import argparse
import re

from loguru import logger
import tensorflow as tf
from lib.dataset.batched_iterator import batched_iterator

from lib.vis.segmentation_plot import segmentation_plot
from lib.io.handle_open import handle_open
from lib.ai.RainfallWaterContraster import RainfallWaterContraster
from lib.dataset.dataset_segmenter import dataset_predict
from lib.io.find_paramsjson import find_paramsjson
from lib.io.readfile import readfile
from lib.vis.segmentation_plot import segmentation_plot


MODE_JSONL = 1
MODE_PNG = 2

def parse_args():
	parser = argparse.ArgumentParser(description="Output water depth image segmentation maps using a given pretrained image segmentation model.")
	# parser.add_argument("--config", "-c", help="Filepath to the TOML config file to load.", required=True)
	parser.add_argument("--input", "-i", help="Path to input directory containing the .tfrecord(.gz) files to predict for. If a single file is passed instead, then only that file will be converted.", required=True)
	parser.add_argument("--output", "-o", help="Path to output file to write output to. If the file extension .png is used instead of .jsonl.gz, then an image is written instead (+d is replaced with the item index).")
	parser.add_argument("--records-per-file", help="Optional, only valid with the .jsonl.gz file extension. If specified, this limits the number of records written to each file. When using this option, you MUST have the string '+d' (without quotes) somewhere in your output filepath.", type=int)
	parser.add_argument("--checkpoint", "-c", help="Checkpoint file to load model weights from.", required=True)
	parser.add_argument("--params", "-p", help="Optional. The file containing the model hyperparameters (usually called 'params.json'). If not specified, it's location will be determined automatically.")
	parser.add_argument("--reads-multiplier", help="Optional. The multiplier for the number of files we should read from at once. Defaults to 0. When using this start with 1.5, which means read ceil(NUMBER_OF_CORES * 1.5). Set to a higher number of systems with high read latency to avoid starving the GPU of data. SETTING THIS WILL SCRAMBLE THE ORDER OF THE DATASET.")
	parser.add_argument("--model-code", help="A description of the model used to predict the data. Will be inserted in the title of png plots.")
	return parser

def run(args):
	
	# Note that we do NOT check to see if the checkpoint file exists, because Tensorflow/Keras requires that we pass the stem instead of the actual index file..... :-/
	
	if (not hasattr(args, "params")) or args.params == None:
		args.params = find_paramsjson(args.checkpoint)
	if (not hasattr(args, "read_multiplier")) or args.read_multiplier == None:
		args.read_multiplier = 0
	if (not hasattr(args, "records_per_file")) or args.records_per_file == None:
		args.records_per_file = 0 # 0 = unlimited
	if (not hasattr(args, "output")) or args.output == None:
		args.output = "-"
	if (not hasattr(args, "model_code")) or args.model_code == None:
		args.model_code = ""
	
	if not os.path.exists(args.params):
		raise Exception(f"Error: The specified filepath params.json hyperparameters ('{args.params}) does not exist.")
	if not os.path.exists(args.checkpoint):
		raise Exception(f"Error: The specified filepath to the checkpoint to load ('{args.checkpoint}) does not exist.")
	
	if args.records_per_file > 0 and args.output.endswith(".jsonl.gz"):
		dirpath_output=os.path.dirname(args.output)
		if not os.path.exists(dirpath_output):
			os.mkdir(dirpath_output)
	
	
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
	
	
	output_mode = MODE_PNG if args.output.endswith(".png") else MODE_JSONL
	logger.info("Output mode is "+("PNG" if output_mode == MODE_PNG else "JSONL"))
	logger.info(f"Records per file: {args.records_per_file}")
	
	if output_mode == MODE_JSONL:
		do_jsonl(args, ai, dataset)
	else:
		do_png(args, ai, dataset, args.model_code)
	
	sys.stderr.write(">>> Complete\n")

def do_png(args, ai, dataset, model_code):
	model_params = json.loads(readfile(args.params))
	
	i = 0
	for rainfall, water in batched_iterator(dataset, tensors_in_item=2, batch_size=model_params["batch_size"]):
		water_predict_batch = ai.embed(rainfall)
		for water_predict in water_predict_batch:
			# [ width, height, softmax_probabilities ] → [ batch, width, height ]
			water_predict = tf.math.argmax(water_predict, axis=-1) 
			# [ width, height ]
			water = tf.squeeze(water)
			
			segmentation_plot(
				water, water_predict,
				model_code,
				args.output.replace("+d", str(i))
			)
			
			i += 1
			
			if i % 100 == 0:
				sys.stderr.write(f"Processed {i} items")

def do_jsonl(args, ai, dataset):
	write_mode = "wt" if args.output.endswith(".gz") else "w"
	
	handle = sys.stdout
	filepath_metadata = None
	if args.output != "-":
		handle = handle_open(
			args.output if args.records_per_file <= 0 else args.output.replace("+d", str(0)),
			write_mode
		)
		filepath_metadata = os.path.join(os.path.dirname(args.output), "metadata.json")
	
	logger.info(f"filepath_output: {args.output}")
	logger.info(f"filepath_params: {filepath_metadata}")
	
	i = 0
	i_file = i
	files_done = 0
	for step_rainfall, step_water in ai.embed(dataset):
		if args.records_per_file > 0 and i_file > args.records_per_file:
			files_done += 1
			i_file = 0
			handle.close()
			logger.info(f"PROGRESS:file {files_done}")
			handle = handle_open(args.output.replace("+d", str(files_done+1)), write_mode)
		
		handle.write(json.dumps(step_rainfall.numpy().tolist(), separators=(',', ':'))+"\n") # Ref https://stackoverflow.com/a/64710892/1460422
		
		if i == 0 or i % 100 == 0:
			sys.stderr.write(f"[pretrain:predict] STEP {i}\r")
		
		i += 1
		i_file += 1

	handle.close()