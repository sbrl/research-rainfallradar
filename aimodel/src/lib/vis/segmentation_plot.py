import matplotlib.pylab as plt

def segmentation_plot(water_actual, water_predict, model_code, filepath_output):
	# water_actual = [ width, height ]
	# water_predict = [ width, height ]
	
	water_actual = water_actual.numpy()
	water_predict = water_predict.numpy()
	
	px = 1 / plt.rcParams['figure.dpi'] # matplotlib sizes are in inches :-( :-( :-(
	width = 768*2
	height = 768
	
	
	plt.rc("font", size=20)
	plt.rc("font", family="Ubuntu")
	figure, axes = plt.subplot_mosaic("AB", figsize=(width*px, height*px))
	
	axes["A"].imshow(water_actual)
	axes["A"].set_title("Actual", fontsize=20)
	
	
	axes["B"].imshow(water_predict)
	axes["B"].set_title("Predicted", fontsize=20)
	
	
	plt.suptitle(f"Rainfall → Water depth prediction | {model_code}", fontsize=28, weight="bold")
	plt.savefig(filepath_output)
	plt.close()