import os

import umap
import umap.plot
import numpy as np
import matplotlib.pylab as plt
import pandas

def vis_embeddings(filepath_output, features):
	dimreducer = umap.UMAP(min_dist=0.05).fit(features)
	
	px = 1 / plt.rcParams['figure.dpi'] # matplotlib sizes are in inches :-( :-( :-(
	width = 1920
	height = 768
	
	plt.rc("font", size=20)
	plt.rc("font", family="Ubuntu")
	figure = plt.figure(figsize=(width*px, height*px))
	figure.add_subplot(1, 2, 1)
	
	# 1: UMAP
	umap.plot.points(dimreducer,
		color_key_cmap="brg", # color_key_cmap="jet",
		ax=figure.get_axes()[0]
	)
	plt.title(f"UMAP Dimensionality Reduction", fontsize=20)
	
	# 2: Parallel coordinates
	figure.add_subplot(1, 2, 2)
	dataframe = pandas.DataFrame(features)
	dataframe["Label"] = [1] * len(features)
	# dataframe["Label"] = range(len(features)) # used when we actually have labels. In this case we don't though
	pandas.plotting.parallel_coordinates(
		dataframe,
		"Label",
		ax=figure.get_axes()[1],
		use_columns=False,
		axvlines=False,
		sort_labels=True
	)
	
	plt.title(f"Parallel coordinates plot", fontsize=20)
	
	plt.suptitle(f"ContrastiveE1 embeddings | ResNetV2 | {len(features)} items", fontsize=28, weight="bold")
	plt.savefig(filepath_output)