
import argparse
import sys
import re
import importlib

# import pysnooper

# @pysnooper.snoop()
def parse_args():
	"""Defines and parses the CLI arguments."""
	
	
	if len(sys.argv) < 2 or sys.argv[1] == "--help":
		sys.stderr.write("""
This program trains, manipulates, visualises, and manages a contrastive learning based rainfall radar → water depth prediction model.
It functions by first finding relationships between the rainfall radar data and the water depth + heightmap data (the 'pretrain' subcommand). After this, a decoder model to predict water depth (modelled as an image segmentation task), can then be trained.

Available subcommands:
	pretrain			Pretrain a contrastive learning model as an encoder.
	pretrain-predict	Make predictions using a trained contrastive learning encoder.
	pretrain-plot		Plot using embeddings predicted using pretrain-predict.

For more information, do src/index.py <subcommand> --help.
""")
		exit(0)
	
	subcommand = re.sub(r'[^a-z0-9-]', '', sys.argv[1])
	
	subcommand_argparser = importlib.import_module(f"subcommands.{subcommand}").parse_args
	
	parser = subcommand_argparser()
	# sys.stderr.write(f"Error: Unknown subcommand '{subcommand} (try --help).\n")
	# exit(1)
	if parser == None:
		sys.stderr.write(f"Error: The subcommand '{subcommand}' did not return an argument parser. This is a bug.\n")
		exit(1)
	
	parser.add_argument("--only-gpu",
		help="If the GPU is not available, exit with an error (useful on shared HPC systems to avoid running out of memory & affecting other users)", action="store_true")
	
	return subcommand, parser.parse_args(args=sys.argv[2:])

