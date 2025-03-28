#!/usr/bin/env python3
import sys
import os
import re
import math
from functools import reduce

import pandas as pd
import holoviews as hv
import datashader as ds
import holoviews.operation.datashader as hvds # REQUIRED, as it's a separate entrypoint for some crazy reason
import holoviews.plotting.plotly
import colorcet as cc
import seaborn as sns

from loguru import logger

import lib.primitives.env as env

hv.extension("matplotlib")

FILEPATH_INPUT = env.read("FILEPATH_INPUT", str, default="-")
FILEPATH_OUTPUT = env.read("FILEPATH_OUTPUT", str)

IMAGE_WIDTH = env.read("IMAGE_WIDTH", int, default=1000)
IMAGE_HEIGHT = env.read("IMAGE_HEIGHT", int, default=1000)

DO_DSRAW = env.read("DO_DSRAW", bool)

WEIGHTING_STAT_BASE = env.read("WEIGHTING_STAT_BASE", str, default="water")
WEIGHTING_RANGE = env.read("WEIGHTING_RANGE", str, default="3500:")
WEIGHTING_RESOLUTION = env.read("WEIGHTING_RESOLUTION", int, default=100)


env.print_all()

# --------------------------------------------


if FILEPATH_INPUT != "-":
	env.val_file_exists(FILEPATH_INPUT)
else:
	FILEPATH_INPUT = sys.stdin

def parse_weighting_range(source):
	if ":" not in source:
		return -math.inf, math.inf
	
	start, end = source.split(":")
	
	start_val = int(start) if start else -math.inf
	end_val = int(end) if end else math.inf
	
	return start_val, end_val

WEIGHTING_RANGE = parse_weighting_range(WEIGHTING_RANGE)

# --------------------------------------------

# compression handled transparently by pandas :D
df = pd.read_csv(FILEPATH_INPUT, sep="\t")


logger.info(f"READ {FILEPATH_INPUT}")

hv.output(backend="plotly")
# if we don't specify this then it will ALWAYS be 400x400 :-(
hv.plotting.plotly.ElementPlot.width = IMAGE_WIDTH
hv.plotting.plotly.ElementPlot.height = IMAGE_HEIGHT


aggregate = ds.Canvas(
	# Reduce the size of the datashader canvas because holoviews is stoopid and will either be a given size or 400x400, and setting holoviews to the same size as the canvas doesn't work bc there's the extra axes etc around the edge to account for
	plot_width=math.floor(IMAGE_WIDTH*0.9),
	plot_height=math.floor(IMAGE_HEIGHT*0.9)
).points(df, y="rainfall", x="water")

img = hvds.shade(hv.Image(aggregate), cmap=cc.b_linear_bmy_10_95_c78)
hv.save(img, FILEPATH_OUTPUT)

if DO_DSRAW:
	# Also plot with raw datashader output (no axes :-() for comparative purposes
	img = ds.tf.set_background(ds.tf.shade(aggregate, cmap=cc.b_linear_bmy_10_95_c78), "white")
	ds.utils.export_image(img, re.sub(r"\.png$", "_dsraw", FILEPATH_OUTPUT))

# plt.imshow(img)
# plt.xlabel('rainfall')
# plt.ylabel('water')
# plt.savefig(re.sub(r"\.png$", "", FILEPATH_OUTPUT) + "_AXES.png")
# plt.close()

logger.success(f"Written graph to {FILEPATH_OUTPUT}")


# ------------------------------------------------------

filepath_output_histogram = re.sub(r"\..*$", f"_hist-{WEIGHTING_STAT_BASE}.png", FILEPATH_OUTPUT)
histogram = sns.histplot(df, x=WEIGHTING_STAT_BASE, element="poly")
histogram.set(
    title=f"{WEIGHTING_STAT_BASE} // {re.sub(r"\..*$", "", os.path.basename(FILEPATH_INPUT))}"
)
print("DEBUG:histogram", histogram)
histogram.figure.savefig(filepath_output_histogram)

logger.success(f"Written histogram for {WEIGHTING_STAT_BASE} to {filepath_output_histogram}")

# ------------------------------------------------------

filepath_output_weightings = re.sub(r"\..*$", f"_WEIGHTS-{WEIGHTING_STAT_BASE}.tsv", FILEPATH_OUTPUT)

stat_ranged = df[
    df[WEIGHTING_STAT_BASE].between(WEIGHTING_RANGE[0], WEIGHTING_RANGE[1])
][WEIGHTING_STAT_BASE].sort_values()


print("DEBUG:stat_ranged SORTED", stat_ranged)

def find_bins(data: pd.Series, bin_count: int):
	"""
	Calculates boundaries & counts for N given bins. This forms the basis of the sample weighting system.
	TODO take advantage of the fact that `data` could be sorted (TODO move sort_values() down into this function?)
	"""
	min_val = data.min()
	max_val = data.max()
	bin_width = (max_val - min_val) / bin_count
	
	thresholds = [float(min_val + (bin_width * i)) for i in range(bin_count + 1)]
	bins = []
	
	for i in range(bin_count):
		lower = thresholds[i]
		upper = thresholds[i + 1]
		count = len(data[(data >= lower) & (data <= upper)])
		bins.append((lower, upper, count))
	
	return bins

def normalise_count(count, ds_min, ds_max):
	ds_range = ds_max - ds_min

	result = (count - ds_min) / ds_range	# Normalise 0..1
	result = 1 - result						# Flip so bins with many values in them are rated lower
	# TODO consider exponential function here?
	# TODO handle really low stuff & filter out like water <= 3500 
	result = 0.1 + (result * 0.9)			# Change to be in range 0.1..1 so that all samples have at least SOME effect
	return result

def normalise_bins(bins):
	count_min = reduce(lambda acc, el: math.min(acc, el[2]))
	count_max = reduce(lambda acc, el: math.max(acc, el[2]))

	# Normalise count to 0..1, also flip such that boxes 
	bins = [(l, u, normalise_count(c, count_min, count_max)) for (l, u, c) in bins]
	bins.insert(0, (-math.inf, bins[0][0], 0.05)) # anything below range is of normalised weight 0.05
	bins.append((bins[-1][1], math.inf, bins[-1][2])) # anything above range takes weighting from the upper bin
	
	return bins

bins = normalise_bins(find_bins(stat_ranged, WEIGHTING_RESOLUTION))

df_bins = pd.DataFrame(bins, columns=["lower", "upper", "weight"])

df_bins.to_csv(filepath_output_weightings, sep="\t")

print("DEBUG:df_bins", df_bins)

logger.success(f"Written weights with {WEIGHTING_RESOLUTION} bins to {filepath_output_weightings}")
