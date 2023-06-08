#!/usr/bin/env python3

import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_metric(ax, train, val, name, dir_output):
	ax.plot(train, label=f"train_{name}")
	ax.plot(val, label=f"val_{name}")
	ax.set_title(name)
	ax.set_xlabel("epoch")
	ax.set_ylabel(name)
	# plt.savefig(os.path.join(dir_output, f"{name}.png"))
	# plt.close()


def plot_metrics(filepath_input, dirpath_output):
	df = pd.read_csv(filepath_input, sep="\t")

	fig = plt.figure(figsize=(10,13))
	for i, colname in enumerate(filter(lambda colname: colname != "epoch" and not colname.startswith("val_"), df.columns.values.tolist())):
		train = df[colname]
		val = df[f"val_{colname}"]
		
		colname_display = colname.replace("metric_dice_coefficient", "dice coefficient") \
			.replace("one_hot_mean_iou", "mean iou")
		
		ax = fig.add_subplot(3, 2, i+1)
		
		plot_metric(ax, train, val, name=colname_display, dir_output=dirpath_output)

	fig.tight_layout()

	target=os.path.join(dirpath_output, f"metrics.png")
	plt.savefig(target)
	
	sys.stderr.write(">>> Saved to ")
	sys.stdout.write(target)
	sys.stderr.write("\n")


if __name__ == "__main__":
	if "INPUT" not in os.environ:
		sys.stderr.write("""
plot_metrics.py: plot metrics for a metrics.tsv file

The output file is named "metrics.png".

Usage:
	INPUT="path/to/metrics.tsv" OUTPUT="path/to/output_dir" path/to/plot_metrics.py 
""")
		sys.exit()
		
	
	FILEPATH_INPUT = os.environ["INPUT"]
	if os.path.isdir(FILEPATH_INPUT):
		FILEPATH_INPUT = os.path.join(FILEPATH_INPUT, "metrics.tsv")
	if not os.path.exists(FILEPATH_INPUT):
		sys.stderr.write(f"Error: The input filepath at {FILEPATH_INPUT} either does not exist ro you don't have permission to read it.\n")
		sys.exit(1)
	DIRPATH_OUTPUT = os.environ["OUTPUT"] if "OUTPUT" in os.environ else os.path.dirname(FILEPATH_INPUT)
	
	plot_metrics(FILEPATH_INPUT, DIRPATH_OUTPUT)
	
