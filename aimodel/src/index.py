#!/usr/bin/env python3

import importlib
import sys
from datetime import datetime

from loguru import logger

def init_logging():
	pass

from parse_args import parse_args

def main():
	subcommand, args = parse_args()
	if args == None:
		return
	
	time_start = datetime.utcnow()
	logger.info(f"Time: Starting subcommand {subcommand} at {str(datetime.utcnow().isoformat())}")
	
	imported_module = importlib.import_module(f"subcommands.{subcommand}")
	imported_module.run(args)
	
	logger.info(f"*** complete in {str((datetime.now() - time_start).total_seconds())} seconds ***")
	
	

if __name__ == "__main__":
    main()
else:
    print("This script must be run directly. It cannot be imported.")
    exit(1)
