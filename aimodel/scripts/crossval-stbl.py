#!/usr/bin/env python3

import os
import sys

from loguru import logger
import pandas as pd
import scipy
from matplotlib import pyplot as plt

# This script analyses metrics.tsv files from a series of identical experiments and reports metrics on them.
# This is sometimes known as cross-validation, but we usually use the model series code crossval-stblX, where X is an integer >0.

ERROR_BAR_MODE = os.environ["ERROR_BAR_MODE"] if "ERROR_BAR_MODE" in os.environ else "stddev"

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

Environment variables:
	ERROR_BAR_MODE=stddev
		Statistic to use for error bars/area. Default: stddev. Possible values: stddev, mad
""")
	sys.exit(0)

DIRPATH = sys.argv[1]  # [0] == script path

files = 0
metrics = {}
epochs = None

for filepath in os.scandir(DIRPATH):
	if not os.path.isdir(filepath):
		continue
	tbl = pd.read_csv(os.path.join(filepath, "metrics.tsv"), sep="\t")

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

logger.info(f"Read {files} files into crossval-stbl{files} analysis")

stats = {}

for metric in metrics.keys():
    metrics[metric] = pd.DataFrame(metrics[metric]).transpose()

    if metric not in stats:
        stats[metric] = {}

    stats[metric]["mad"] = scipy.stats.median_abs_deviation(
        metrics[metric], axis=1
    )  # median absolute deviation
    stats[metric]["stddev"] = metrics[metric].std(axis=1)
    stats[metric]["mean"] = metrics[metric].mean(axis=1)
    stats[metric]["min"] = metrics[metric].min(axis=1)
    stats[metric]["max"] = metrics[metric].max(axis=1)
    stats[metric]["agg_min"] = stats[metric]["min"].min()
    stats[metric]["agg_max"] = stats[metric]["max"].max()
    stats[metric]["agg_stddev"] = metrics[metric].stack().std()
    stats[metric]["agg_mean"] = metrics[metric].stack().std()
    stats[metric]["agg_mad"] = scipy.stats.median_abs_deviation(
        metrics[metric].stack()
    )  # median absolute deviation

    # print(stats[metric])

    plt.figure(figsize=(12, 8))
    plt.ylim(min(0, stats[metric]["agg_min"]), max(1, stats[metric]["agg_max"]))
    plt.grid(visible=True, which="major", axis="y", linewidth=2)
    plt.grid(visible=True, which="minor", axis="y", linewidth=1)
    plt.minorticks_on()
    plt.rcParams['font.size'] = 32
    plt.fill_between(
        epochs,
        stats[metric]["min"],
        stats[metric]["max"],
        alpha=0.2,
        facecolor="#B7DE28",
        edgecolor="#FDE724",
        linestyle="dotted",
        linewidth=1,
    )
    plt.fill_between(
        epochs,
        stats[metric]["mean"] - stats[metric][ERROR_BAR_MODE],
        stats[metric]["mean"] + stats[metric][ERROR_BAR_MODE],
        alpha=0.5,
        facecolor="#228A8D",
        edgecolor="#3CBB74",
        linestyle="dashed",
        linewidth=1,
    )
    plt.plot(epochs, stats[metric]["mean"], color="#450C54")
    plt.title(f"{metric} // crossval-stbl{files}")
    plt.xlabel("epoch")
    plt.ylabel(metric)
    plt.savefig(os.path.join(DIRPATH, f"crossval-stbl{files}_{metric}_{ERROR_BAR_MODE}.png"))
    plt.close()

logger.success(f"Written {len(stats.keys())} graphs to {DIRPATH}")
