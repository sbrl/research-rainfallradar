import math

from loguru import logger
import tensorflow as tf

from .components.convnext import make_convnext
from .components.convnext_inverse import do_convnext_inverse
from .components.LayerStack2Image import LayerStack2Image

def model_rainfallwater_mono(metadata, shape_water_out, model_arch_enc="convnext_xtiny", model_arch_dec="convnext_i_xtiny", feature_dim=512, batch_size=64, water_bins=2):
	"""Makes a new rainfall / waterdepth mono model.

	Args:
		metadata (dict): A dictionary of metadata about the dataset to use to build the model with.
		shape_water_out (int[]): The width and height (in that order) that should dictate the output shape of the segmentation head. CURRENTLY NOT USED.
		model_arch (str, optional): The architecture code for the underlying (inverted) ConvNeXt model. Defaults to "convnext_i_xtiny".
		batch_size (int, optional): The batch size. Reduce to save memory. Defaults to 64.
		water_bins (int, optional): The number of classes that the water depth output oft he segmentation head should be binned into. Defaults to 2.

	Returns:
		tf.keras.Model: The new model, freshly compiled for your convenience! :D
	"""
	rainfall_channels, rainfall_width, rainfall_height = metadata["rainfallradar"] # shape = [channels, width, height]
	
	print("RAINFALL channels", rainfall_channels, "width", rainfall_width, "height", rainfall_height)
	out_water_width, out_water_height = shape_water_out
	
	layer_input = tf.keras.layers.Input(
		shape=(rainfall_width, rainfall_height, rainfall_channels)
	)
	
	# ENCODER
	layer_next = make_convnext(
		input_shape				= (rainfall_width, rainfall_height, rainfall_channels),
		classifier_activation	= tf.nn.relu, # this is not actually a classifier, but rather a feature encoder
		num_classes				= feature_dim, # size of the feature dimension, see the line above this one
		arch_name				= model_arch_enc
	)(layer_input)
	
	print("ENCODER output_shape", layer_next.shape)
	
	# BOTTLENECK
	layer_next = tf.keras.layers.Dense(name="cns.stage.bottleneck.dense2", units=feature_dim)(layer_input)
	layer_next = tf.keras.layers.Activation(name="cns.stage.bottleneck.gelu2", activation="gelu")(layer_next)
	layer_next = tf.keras.layers.LayerNormalization(name="cns.stage.bottleneck.norm2", epsilon=1e-6)(layer_next)
	layer_next = tf.keras.layers.Dropout(name="cns.stage.bottleneck.dropout", rate=0.1)(layer_next)
	
	# DECODER
	layer_next = LayerStack2Image(target_width=4, target_height=4)(layer_next)
	# layer_next = tf.keras.layers.Reshape((4, 4, math.floor(feature_dim_in/(4*4))), name="cns.stable_begin.reshape")(layer_next)
	
	layer_next = tf.keras.layers.Dense(name="cns.stage.begin.dense2", units=feature_dim)(layer_next)
	layer_next = tf.keras.layers.Activation(name="cns.stage_begin.relu2", activation="gelu")(layer_next)
	layer_next = tf.keras.layers.LayerNormalization(name="cns.stage_begin.norm2", epsilon=1e-6)(layer_next)
	
	layer_next = do_convnext_inverse(layer_next, arch_name=model_arch_dec)
	
	# TODO: An attention layer here instead of a dense layer, with a skip connection perhaps?
	logger.warning("Warning: TODO implement attention from https://ieeexplore.ieee.org/document/9076883")
	layer_next = tf.keras.layers.Dense(32, activation="gelu")(layer_next)
	layer_next = tf.keras.layers.Conv2D(water_bins, activation="gelu", kernel_size=1, padding="same")(layer_next)
	layer_next = tf.keras.layers.Softmax(axis=-1)(layer_next)
	
	model = tf.keras.Model(
		inputs = layer_input,
		outputs = layer_next
	)
	
	model.compile(
		optimizer="Adam",
		loss=tf.keras.losses.CategoricalCrossentropy(),
		metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
	)
	
	return model