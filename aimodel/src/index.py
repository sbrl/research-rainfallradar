#!/usr/bin/env python3

import importlib
import sys

from loguru import logger

def init_logging():
	pass

from parse_args import parse_args

def main():
	subcommand, args = parse_args()
	if args == None:
		return
	
	imported_module = importlib.import_module(f"subcommands.{subcommand}")
	# TODO: Support multiple subcommands here
	match subcommand:
		case "pretrain":
			imported_module.pretrain(args)
		case _:
			sys.stderr.write(f"Error: The subcommand '{subcommand}' hasn't been registered in index.py yet.\nThis is a bug.\n")
			exit(1)
	
	

if __name__ == "__main__":
    main()
else:
    print("This script must be run directly. It cannot be imported.")
    exit(1)
