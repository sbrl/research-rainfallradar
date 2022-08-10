#!/usr/bin/env python3
import sys
import os
import math
import gzip
import json
import argparse

from silence_tensorflow import silence_tensorflow
if not os.environ.get("NO_SILENCE"):
	silence_tensorflow()
import tensorflow as tf

def parse_args():
	parser = argparse.ArgumentParser(description="Convert a generated .jsonl.gz file to a .tfrecord.gz file")
	parser.add_argument("--input", "-i", help="Path to the input file to convert.", required=True)
	parser.add_argument("--output", "-o", help="Path to the output file to write to.", required=True)
	return parser.parse_args(args=sys.argv[1:])

def convert(filepath_in, filepath_out):
	options = tf.io.TFRecordOptions(compression_type="GZIP", compression_level=9)
	with gzip.open(filepath_in, "r") as handle, tf.io.TFRecordWriter(filepath_out, options=options) as writer:
		i = -1
		for line in handle:
			i += 1
			if len(line) == 0:
				continue
			
			###
			## 1: Parse JSON
			###
			obj = json.loads(line)
			
			###
			## 2: Convert to tensor
			###
			rainfall = tf.constant(obj["rainfallradar"], dtype=tf.float32)
			water = tf.constant(obj["waterdepth"], dtype=tf.float32)
			
			###
			## 3: Print shape definitions (required when parsing)
			###
			if i == 0:
				print("SHAPES\t"+json.dumps({ "rainfallradar": rainfall.shape.as_list(), "waterdepth": water.shape.as_list() }), flush=True)
			
			###
			## 4: Serialise tensors
			###
			rainfall = tf.train.BytesList(value=[tf.io.serialize_tensor(rainfall, name="rainfall").numpy()])
			water = tf.train.BytesList(value=[tf.io.serialize_tensor(water, name="water").numpy()])
			
			###
			## 5: Write to .tfrecord.gz file
			###
			record = tf.train.Example(features=tf.train.Features(feature={
				"rainfallradar": tf.train.Feature(bytes_list=rainfall),
				"waterdepth": tf.train.Feature(bytes_list=water)
			}))
			writer.write(record.SerializeToString())
			
			print(f"{i}", flush=True)
			


def main():
	args = parse_args()
	
	if not os.path.exists(args.input):
		print(f"Error: No such input file {args.input}")
		sys.exit(2)
	
	
	convert(args.input, args.output)

if __name__ == "__main__":
    main()
else:
    print("This script must be run directly. It cannot be imported.")
    exit(1)
