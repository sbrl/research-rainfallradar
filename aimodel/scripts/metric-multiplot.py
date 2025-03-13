#!/usr/bin/env python3

import os
import sys
import re

from loguru import logger
import pandas as pd
from matplotlib import pyplot as plt

# This script analyses metrics.tsv files from a series of experiments from the same codebase and reports metrics on them.
# In other words, it plots them all on the same graph for comparison purposes - e.g. inter-model within a given series.

if len(sys.argv) <= 1:
	print("""
Usage:
	scripts/metric-multiplot.mjs {{path/to/directory}}

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
DIRPATH_basename = os.path.basename(DIRPATH)

files = 0
metrics = {}
epochs = None

model_codes = []

for filepath in os.scandir(DIRPATH):
    if not os.path.isdir(filepath):
        continue
    tbl = pd.read_csv(os.path.join(filepath, "metrics.tsv"), sep="\t")

    model_code = re.sub(r"^.*-", "", os.path.basename(filepath))
    model_codes.append(model_code)

    logger.info(f"Found model_code {model_code} -> {os.path.basename(filepath)}")

    # metrics.append(tbl)

    for column in tbl.columns:
        if column == "epoch":
            if epochs is None:
                epochs = tbl[column]
            continue  # Row index implicitly retains this

        if column not in metrics:
            metrics[column] = []

        metrics[column].append(tbl[column].values)
        # print(column, tbl[column])

    # print("DEBUG:metrics", tbl)
    files += 1

logger.info(f"Read {files} files into {files}-way multiplot")


i = 0
for metric in metrics.keys():
	# print(stats[metric])

	filepath_graph_next = os.path.join(DIRPATH, f"metric-multiplot{files}_{metric}.png")

	plt.figure(figsize=(12, 8))
	plt.rcParams["font.size"] = 22
	# plt.ylim(min(0, stats[metric]["agg_min"]), max(1, stats[metric]["agg_max"]))
	plt.grid(visible=True, which="major", axis="y", linewidth=2)
	plt.grid(visible=True, which="minor", axis="y", linewidth=1)
	plt.minorticks_on()
	colnum = 0
	for col in metrics[metric]:
		plt.plot(epochs, col, label=model_codes[colnum])
		colnum += 1
	plt.legend(model_codes, fontsize=14, title_fontsize=16)
	plt.title(f"{metric} // {DIRPATH_basename} // metric-multiplot{files}")
	plt.xlabel("epoch")
	plt.ylabel(metric)
	plt.savefig(filepath_graph_next)
	plt.close()
	logger.success(f"GRAPH OUT {filepath_graph_next}")
	
	i += 1

logger.success(f"Written {i} graphs to {DIRPATH}")
