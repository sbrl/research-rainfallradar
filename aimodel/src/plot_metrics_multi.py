#!/usr/bin/env python3

import sys
import os
import re
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

def do_regex(source, regex):
	if regex is None or len(regex) == 0:
		return source
	
	result = re.search(regex, source)
	if not result:
		return source
	return result.group(0)

def plot_metric(ax, train_list, val_list, metric_name, model_names, dir_output):
	i = 0
	for train in train_list:
		ax.plot(train, label=model_names[i], linewidth=1)
		i += 1
	i = 0
	for val in val_list:
		ax.plot(val, label=f"val_{model_names[i]}", linewidth=1)
		i += 1
	
	ax.set_title(metric_name)
	ax.set_xlabel("epoch")
	ax.set_ylabel(metric_name)
	# plt.savefig(os.path.join(dir_output, f"{name}.png"))
	# plt.close()

def make_dfs(filepaths_input):
	dfs = []
	for filepath_input in filepaths_input:
		print("DEBUG filepath_input", filepath_input)
		dfs = pd.read_csv(filepath_input, sep="\t")

def plot_metrics(filepaths_input, model_names, dirpath_output, resolution=1):
	dfs = [ pd.read_csv(filepath_input, sep="\t") for filepath_input in filepaths_input ]
	matplotlib.rcParams.update({'font.size': 15*resolution})
	fig = plt.figure(figsize=(10*resolution, 14*resolution))
	for i, colname in enumerate(filter(lambda colname: colname != "epoch" and not colname.startswith("val_"), dfs[0].columns.values.tolist())):
		train = [ df[colname] for df in dfs ]
		val = [ df[f"val_{colname}"] for df in dfs ]
		
		colname_display = colname.replace("metric_dice_coefficient", "dice coefficient") \
			.replace("one_hot_mean_iou", "mean iou")
		
		ax = fig.add_subplot(3, 2, i+1)
		
		plot_metric(ax, train, val, metric_name=colname_display, model_names=model_names, dir_output=dirpath_output)
	
	
	# Ref https://stackoverflow.com/a/57484812/1460422
	# lines_labels = [ ax.get_legend_handles_labels() for ax in fig.axes ]
	lines_labels = [ fig.axes[0].get_legend_handles_labels() ]
	lines, labels = [sum(lol, []) for lol in zip(*lines_labels) ]
	legend = fig.legend(lines, labels, loc='outside upper center', ncol=3)
	
	# Ref https://stackoverflow.com/a/48296983/1460422
	# change the line width for the legend
	for line in legend.get_lines():
		line.set_linewidth(4.0*resolution)
	
	fig.tight_layout()
	plt.subplots_adjust(top=0.85)
	
	target=os.path.join(dirpath_output, f"metrics.png")
	plt.savefig(target, bbox_inches='tight')
	
	sys.stderr.write(">>> Saved to ")
	sys.stdout.write(target)
	sys.stderr.flush(); sys.stdout.flush()
	sys.stderr.write("\n")


if __name__ == "__main__":
	if "--help" in sys.argv:
		sys.stderr.write("""
plot_metrics_multi.py: plot metrics for more than one metrics.tsv file

It is assumed that all files have identical metrics in the same column order.

The output file is named "metrics.png".

Usage:
	echo -e "filepathA\\nfilepathB..." | [OUTPUT="path/to/output_dir"] [REGEX_NAME=''] path/to/plot_metrics_multi.py
""")
		sys.exit()
	
	REGEX_NAME = os.environ["REGEX_NAME"] if "REGEX_NAME" in os.environ else None
	if REGEX_NAME is None and len(sys.argv) >= 1:
		REGEX_NAME = sys.argv[1]
	FILEPATHS_INPUT = []
	MODEL_NAMES = []
	for line in sys.stdin:
		filepath = line
		if not os.path.exists(filepath):
			filepath = filepath.strip()
		if os.path.isdir(filepath):
			filepath = os.path.join(filepath, "metrics.tsv")
		if not os.path.exists(filepath):
			sys.stderr.write(f"Warning: The input filepath at {filepath} either does not exist or you don't have permission to read it.\n")
		
		
		FILEPATHS_INPUT.append(filepath)
		
		stem = os.path.basename(os.path.dirname(filepath))
		MODEL_NAMES.append(do_regex(stem, REGEX_NAME) if REGEX_NAME is not None and len(REGEX_NAME) > 0 else stem)
		
	
	
	sys.stderr.write(">>> MAPPING:\n")
	i = 0
	for model_name in MODEL_NAMES:
		sys.stderr.write(f"    {model_name} -- {FILEPATHS_INPUT[i]}\n")
		i += 1
	
	DIRPATH_OUTPUT = os.environ["OUTPUT"] if "OUTPUT" in os.environ else os.getcwd()
	
	plot_metrics(
		FILEPATHS_INPUT,
		MODEL_NAMES,
		DIRPATH_OUTPUT,
		resolution=float(os.environ["RESOLUTION"]) if "RESOLUTION" in os.environ else 1
	)
	
