import os

import umap
import umap.plot
import numpy as np
import matplotlib.pylab as plt
import pandas

def vis_embeddings(filepath_output, features):
	dimreducer = umap.UMAP(
		# min_dist=0.05
	).fit(features)
	
	px = 1 / plt.rcParams['figure.dpi'] # matplotlib sizes are in inches :-( :-( :-(
	width = 8000
	height = 768
	
	plt.rc("font", size=20)
	plt.rc("font", family="Ubuntu")
	figure, axes = plt.subplot_mosaic("ABBBB", figsize=(width*px, height*px))
	# figure.add_subplot(1, 2, 1)
	
	# 1: UMAP
	umap.plot.points(dimreducer,
		ax=axes["A"]
	)
	axes["A"].set_title(f"UMAP Dimensionality Reduction", fontsize=20)
	
	# 2: Parallel coordinates
	dataframe = pandas.DataFrame(features)
	dataframe["Label"] = [1] * len(features)
	# dataframe["Label"] = range(len(features)) # used when we actually have labels. In this case we don't though
	pandas.plotting.parallel_coordinates(
		dataframe,
		"Label",
		ax=axes["B"],
		use_columns=False,
		axvlines=False,
		sort_labels=True
	)
	
	axes["B"].set_title(f"Parallel coordinates plot", fontsize=20)
	
	plt.suptitle(f"RainfallContrastive embeddings | rainfall | E2 ConvNeXt | {len(features)} items", fontsize=28, weight="bold")
	plt.savefig(filepath_output)
	plt.close()