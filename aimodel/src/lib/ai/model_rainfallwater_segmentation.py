import math

from loguru import logger
import tensorflow as tf

from .components.convnext_inverse import do_convnext_inverse


def model_rainfallwater_segmentation(metadata, shape_water_out, model_arch="convnext_i_xtiny", batch_size=64, summary_file=None, water_bins=2):
	"""Makes a new rainfall / waterdepth segmentation head model.

	Args:
		metadata (dict): A dictionary of metadata about the dataset to use to build the model with.
		feature_dim_in (int): The size of the feature dimension 
		shape_water_out (_type_): _description_
		model_arch (str, optional): _description_. Defaults to "convnext_i_xtiny".
		batch_size (int, optional): _description_. Defaults to 64.
		summary_file (_type_, optional): _description_. Defaults to None.
		water_bins (int, optional): _description_. Defaults to 2.

	Returns:
		_type_: _description_
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