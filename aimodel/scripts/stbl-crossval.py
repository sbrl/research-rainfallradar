#!/usr/bin/env python3

import os
import sys

from loguru import logger
import pandas as pd

# This script analyses metrics.tsv files from a series of identical experiments and reports metrics on them.
# This is sometimes known as cross-validation, but we usually use the model series code crossval-stblX, where X is an integer >0.


if len(sys.argv) <= 1:
	print("""
Usage:
	scripts/stbl-crossval.mjs {{path/to/directory}}

...in which the given directory contains a series of experiment root directories to include in the statistical analysis.

This script is not picky about the format of the data in metrics.tsv, so long as it's in the form:

epoch	metric_A	metric_B	…
0	val:float	val:float	…
1	val:float	val:float	…
2	val:float	val:float	…
⋮
""")
	sys.exit(0)

DIRPATH = sys.argv[1]  # [0] == script path

files = 0
metrics = {}

for filepath in os.scandir(DIRPATH):
	tbl = pd.read_csv(os.path.join(filepath, "metrics.tsv"), sep="\t")

	# metrics.append(tbl)

	for column in tbl.columns:
		if column == "epoch":
			continue  # Row index implicitly retains this

		if column not in metrics:
			metrics[column] = []

		metrics[column].append(tbl[column].values)
		# print(column, tbl[column])

	# print("DEBUG:metrics", tbl)
	files += 1

logger.info(f"Read {files} files into crossval-stbl{files} analysis")

stats = {}

for metric in metrics.keys():
	metrics[metric] = pd.DataFrame(metrics[metric]).transpose()

	if metric not in stats:
		stats[metric] = {}

	stats[metric]["aad"] = metrics[metric].max(axis=1) # mean/average absolute deviation
	stats[metric]["mad"] = metrics[metric].max(axis=1) # median absolute deviation
	stats[metric]["stddev"] = metrics[metric].std(axis=1)
	stats[metric]["mean"] = metrics[metric].mean(axis=1)
	stats[metric]["min"] = metrics[metric].min(axis=1)
	stats[metric]["max"] = metrics[metric].max(axis=1)
	stats[metric]["agg_min"] = stats[metric]["min"].min()
	stats[metric]["agg_max"] = stats[metric]["max"].max()
	stats[metric]["agg_stddev"] = metrics[metric].stack().std()
	stats[metric]["agg_mean"] = metrics[metric].stack().std()
	stats[metric]["agg_aad"] = metrics[metric].stack().max() # mean/average absolute deviation

	print(stats[metric])

