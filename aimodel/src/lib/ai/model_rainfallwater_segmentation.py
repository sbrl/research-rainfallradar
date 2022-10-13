import math

from loguru import logger
import tensorflow as tf

from .components.convnext_inverse import do_convnext_inverse


def model_rainfallwater_segmentation(metadata, shape_water_out, model_arch="convnext_i_xtiny", batch_size=64, water_bins=2):
	"""Makes a new rainfall / waterdepth segmentation head model.

	Args:
		metadata (dict): A dictionary of metadata about the dataset to use to build the model with.
		shape_water_out (int[]): The width and height (in that order) that should dictate the output shape of the segmentation head. CURRENTLY NOT USED.
		model_arch (str, optional): The architecture code for the underlying (inverted) ConvNeXt model. Defaults to "convnext_i_xtiny".
		batch_size (int, optional): The batch size. Reduce to save memory. Defaults to 64.
		water_bins (int, optional): The number of classes that the water depth output oft he segmentation head should be binned into. Defaults to 2.

	Returns:
		tf.keras.Model: The new model, freshly compiled for your convenience! :D
	"""
	out_water_width, out_water_height = shape_water_out
	feature_dim_in = metadata["rainfallradar"][0]
	
	layer_input = tf.keras.layers.Input(
		shape=(feature_dim_in)
	)
	
	# BEGIN
	layer_next = tf.keras.layers.Dense(name="cns.stage.begin.dense1", units=feature_dim_in)(layer_input)
	layer_next = tf.keras.layers.ReLU(name="cns.stage_begin.relu1")(layer_next)
	layer_next = tf.keras.layers.LayerNormalization(name="cns.stage_begin.norm1", epsilon=1e-6)(layer_next)
	
	layer_next = tf.keras.layers.Reshape((4, 4, math.floor(feature_dim_in/(4*4))), name="cns.stable_begin.reshape")(layer_next)
	layer_next = tf.keras.layers.Dense(name="cns.stage.begin.dense2", units=feature_dim_in)(layer_next)
	layer_next = tf.keras.layers.ReLU(name="cns.stage_begin.relu2")(layer_next)
	layer_next = tf.keras.layers.LayerNormalization(name="cns.stage_begin.norm2", epsilon=1e-6)(layer_next)
	
	
	# layer_next = tf.keras.layers.Reshape((1, 1, feature_dim_in), name="cns.stable_begin.reshape")(layer_next)
	
	layer_next = do_convnext_inverse(layer_next, arch_name=model_arch)
	
	# TODO: An attention layer here instead of a dense layer, with a skip connection perhaps?
	logger.warning("Warning: TODO implement attention from https://ieeexplore.ieee.org/document/9076883")
	layer_next = tf.keras.layers.Dense(32)(layer_next)
	layer_next = tf.keras.layers.Conv2D(water_bins, kernel_size=1, activation="softmax", padding="same")(layer_next)
	
	model = tf.keras.Model(
		inputs = layer_input,
		outputs = layer_next
	)
	
	model.compile(
		optimizer="Adam",
		loss=tf.keras.losses.SparseCategoricalCrossentropy(),
		metrics=["accuracy"]
	)
	
	return model