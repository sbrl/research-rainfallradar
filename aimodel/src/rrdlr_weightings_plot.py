#!/usr/bin/env python3
import platform
import sys
import os
import re
import math
from functools import reduce

import numpy as np
import pandas as pd
import holoviews as hv
import datashader as ds
import holoviews.operation.datashader as hvds # REQUIRED, as it's a separate entrypoint for some crazy reason
import holoviews.plotting.plotly
import colorcet as cc
import seaborn as sns
from matplotlib import pyplot as plt

from loguru import logger

import lib.primitives.env as env

hv.extension("matplotlib")

DIRPATH_RAINFALLWATER = env.read("DIRPATH_RAINFALLWATER", str, None)	# Specify to call the summation script rrdlr_calculate_sums automatically

FILEPATH_INPUT = env.read("FILEPATH_INPUT", str, default="-")	# Input summation file, if the above is specified this file is generated automatically, defaults to stdin
FILEPATH_OUTPUT = env.read("FILEPATH_OUTPUT", str)	# Filepath to output for plots. Variations of this filepath are used for other related outputs

IMAGE_WIDTH = env.read("IMAGE_WIDTH", int, default=1000)	# Width of the water vs rainfall plots
IMAGE_HEIGHT = env.read("IMAGE_HEIGHT", int, default=1000)	# Height of the water vs rainfall plots

DO_DSRAW = env.read("DO_DSRAW", bool)	# Also plot with raw datashader & save alongside, default false

WEIGHTING_STAT_BASE = env.read("WEIGHTING_STAT_BASE", str, default="water")	# The stat to use to calculate loss weightings
WEIGHTING_RANGE = env.read("WEIGHTING_RANGE", str, default="3500:")	# Limit the range of values used for the given stat when calculating loss weightings
WEIGHTING_RESOLUTION = env.read("WEIGHTING_RESOLUTION", int, default=100)	# The # of bins to split the stat into


env.print_all()

# --------------------------------------------

if DIRPATH_RAINFALLWATER is not None and os.path.isdir(DIRPATH_RAINFALLWATER):
	from rrdlr_calculate_sums import do_calculate_sums
	
	logger.info("***** CALLING SUBSCRIPT CALCULATE_SUMS *****")
	os.environ["FILEPATH_OUTPUT"] = FILEPATH_INPUT # we've read the env vars above now anyway, so we can fiddle this for the subscript relatively easily here
	do_calculate_sums()

# --------------------------------------------

logger.info(f"Loss weightings calculation and plotting begins! I am python {platform.python_version()}.")

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


logger.info("Binning and calculating weightings")

def find_bins(data: pd.Series, bin_count: int):
	"""
	Calculates boundaries & counts for N given bins. This forms the basis of the sample weighting system.
	TODO take advantage of the fact that `data` could be sorted (TODO move sort_values() down into this function?)
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

def normalise_bin_counts(bins):
	count_min = reduce(lambda acc, el: min(acc, el[2]), bins, math.inf)
	count_max = reduce(lambda acc, el: max(acc, el[2]), bins, -math.inf)

	# Normalise count to 0..1, also flip such that boxes 
	bins = [(lwr, upr, normalise_count(count, count_min, count_max)) for (lwr, upr, count) in bins]
	bins.insert(0, (-math.inf, bins[0][0], 0.05)) # anything below range is of normalised weight 0.05
	bins.append((bins[-1][1], math.inf, bins[-1][2])) # anything above range takes weighting from the upper bin
	
	return bins


bins = normalise_bin_counts(find_bins(stat_ranged, WEIGHTING_RESOLUTION))

df_bins = pd.DataFrame(bins, columns=["lower", "upper", "weight"])

df_bins.to_csv(filepath_output_weightings, sep="\t")

bound_min = float(df[WEIGHTING_STAT_BASE].min())
bound_max = float(df[WEIGHTING_STAT_BASE].max())

# --------------

filepath_output_histogram_bins = re.sub(r"\..*$", f"_hist-{WEIGHTING_STAT_BASE}_weights-norm.png", FILEPATH_OUTPUT)

# Calculations for drawing the histogram thata re WAY too complicated for what they are... ugh >_<
hist_counts = [count for (_, _, count) in bins]
hist_bins = (
    [bins[1][1] - ((bins[2][1] - bins[1][1]) * 4)]
    + [lwr for (lwr, upr, count) in bins[1:-1]]
    + [bound_max]
)
diff = np.diff(hist_bins).tolist()
diff.append(diff[1]) # Last bin is 'everything higher than this', just as the 1st is 'everything lower than this'

# Ref https://stackoverflow.com/a/72072161/1460422
fig, ax = plt.subplots(figsize=(16, 12))
ax.bar(x=hist_bins, height=hist_counts, width=diff, align="edge")
ax.set_xlabel(f"{WEIGHTING_STAT_BASE}")
ax.set_ylabel("normalised count")
ax.set_title(f"weightings: {WEIGHTING_STAT_BASE} // normalised counts {re.sub(r"\..*$", "", os.path.basename(FILEPATH_INPUT))}")
ax.margins(x=0)
plt.tight_layout()
plt.savefig(filepath_output_histogram_bins)

# histplot_bins = sns.barplot(hist_df,
# 	x="bins", y="weights"
# 	# bins = hist_bins,
# 	# weights = hist_weights
# )
# histplot_bins.set(
# 	title=f"weightings: {WEIGHTING_STAT_BASE} // normalised counts {re.sub(r"\..*$", "", os.path.basename(FILEPATH_INPUT))}",
# 	xlabel=f"{WEIGHTING_STAT_BASE}",
# 	ylabel="normalised count",
# )
# histplot_bins.tick_params(
# 	labelrotation=90
# )
# histplot_bins.figure.savefig(filepath_output_histogram_bins)

logger.success(f"Written normalised weights histogram plot to {filepath_output_histogram_bins}")

# --------------


print("DEBUG:df_bins", df_bins)

logger.success(f"Written weights with {WEIGHTING_RESOLUTION} bins to {filepath_output_weightings}")
