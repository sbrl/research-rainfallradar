#!/usr/bin/env python3
import sys

from datetime import datetime
from loguru import logger

import tensorflow as tf
from tqdm import tqdm


import lib.primitives.env as env
from lib.dataset.dataset_mono import dataset_mono_predict  # Because we want the whole dataset at once, not split train/test/silly
from lib.io.handle_open import handle_open

# --------------------------------

def do_calculate_sums():
	time_start = datetime.now()
	logger.info(f"Starting at {str(datetime.now().isoformat())}")
	logger.info(f"I, Tensorflow am version {tf.__version__}")

	# --------------------------------

	FILEPATH_OUTPUT = env.read("FILEPATH_OUTPUT", str, default="-")  # e.g. .tsv.gz
	DIRPATH_RAINFALLWATER = env.read("DIRPATH_RAINFALLWATER", str)
	env.val_dir_exists(DIRPATH_RAINFALLWATER)
	BATCH_SIZE = env.read("BATCH_SIZE", int, default=64)
	WATER_THRESHOLD = env.read("WATER_THRESHOLD", float, default=0.1)
	PARALLEL_READS = env.read("PARALLEL_READS", float, default=1.5) # Set to False to preserve input order


	env.print_all()

	# --------------------------------

	handle_out = sys.stdout if FILEPATH_OUTPUT == "-" else handle_open(FILEPATH_OUTPUT, mode="w", force_textwrite_gzip=True)

	ds = dataset_mono_predict(
		DIRPATH_RAINFALLWATER,
		batch_size=BATCH_SIZE,
		water_threshold=WATER_THRESHOLD,
		rainfall_scale_up=2, # otherwise it won't match the water depth map :-(
		# ri is the default
		output_size=128,  # highest value that fits the data - input_size is "same"
		parallel_reads_multiplier=PARALLEL_READS,
		shuffle=False if PARALLEL_READS == 0 else True,
	)

	handle_out.write("rainfall\twater\n".encode("utf-8"))

	for (batch_rainfall, batch_water) in tqdm(ds, unit="batches"):
		# scale it back down again 'cause we need to scale up to avoid crashing
		batch_rainfall = tf.nn.max_pool(batch_rainfall, ksize=2, strides=2, padding="SAME")
		
		
		count_rainfall = tf.math.reduce_sum(batch_rainfall, axis=range(1, len(batch_rainfall.shape)))
		count_water = tf.math.reduce_sum(batch_water, axis=range(1, len(batch_water.shape)))
		
		counts = tf.transpose(tf.stack([count_rainfall, count_water])).numpy().tolist()
		
		counts = "\n".join([ "\t".join([str(cell) for cell in row]) for row in counts ])
		handle_out.write((counts + "\n").encode("utf-8"))


	handle_out.close()
	
	logger.success(f"Done in {str(datetime.now() - time_start)}, output written to {FILEPATH_OUTPUT}")


if __name__ == "__main__":
	do_calculate_sums()

