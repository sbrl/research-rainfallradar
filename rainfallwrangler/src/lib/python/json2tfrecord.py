import sys
import os
import gzip
import json
import argparse

import tensorflow as tf

def parse_args():
	parser = argparse.ArgumentParser(description="Convert a generated .jsonl.gz file to a .tfrecord.gz file")
	parser.add_argument("--input", "-i", help="Path to the input file to convert.", required=True)
	parser.add_argument("--output", "-o", help="Path to the output file to write to.", required=True)
	
	return parser.parse_args(args=sys.argv[2:])

def convert(filepath_in, filepath_out):
	with gzip.open(filepath_in, "r") as handle, tf.io.TFRecordWriter(filepath_out) as writer:
		for line in handle:
			if len(line) == 0:
				continue
			
			obj = json.loads(line)
			
			rainfall = tf.constant(obj.rainfallradar, dtype=tf.float32)
			water = tf.constant(obj.waterdepth, dtype=tf.float32)
			
			record = tf.train.Example(features=tf.train.Features(feature={
				"rainfallradar": tf.train.BytesList(bytes_list=tf.io.serialize_tensor(rainfall)),
				"waterdepth": tf.train.BytesList(bytes_list=tf.io.serialize_tensor(water))
			}))
			writer.write(record.SerializeToString())


def main():
	args = parse_args()
	
	if not os.path.exists(args.input):
		print(f"Error: No such input file {args.input}")
		sys.exit(2)
	
	
	convert(args.input, args.output)