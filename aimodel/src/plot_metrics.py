#!/usr/bin/env python3

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

FILEPATH_INPUT = os.environ["INPUT"]
DIRPATH_OUTPUT = os.environ["OUTPUT"] if "OUTPUT" in os.environ else os.getcwd()


df = pd.read_csv(FILEPATH_INPUT, sep="\t")

fig = plt.figure(figsize=(10,13))
for i, colname in enumerate(filter(lambda colname: colname != "epoch" and not colname.startswith("val_"), df.columns.values.tolist())):
	train = df[colname]
	val = df[f"val_{colname}"]
	
	colname_display = colname.replace("metric_dice_coefficient", "dice coefficient") \
		.replace("one_hot_mean_iou", "mean iou")
	
	ax = fig.add_subplot(3, 2, i+1)
	
	plot_metric(ax, train, val, name=colname_display, dir_output=DIRPATH_OUTPUT)

fig.tight_layout()

target=os.path.join(DIRPATH_OUTPUT, f"metrics.png")
plt.savefig(target)

print(f">>> Saved to {target}")