import math

from loguru import logger
import tensorflow as tf

from .components.convnext import make_convnext
from .components.convnext_inverse import do_convnext_inverse
from .components.LayerStack2Image import LayerStack2Image
from .components.LossCrossentropy import LossCrossentropy

def model_rainfallwater_mono(metadata, model_arch_enc="convnext_xtiny", model_arch_dec="convnext_i_xtiny", feature_dim=512, batch_size=64, water_bins=2, learning_rate=None, heightmap_input=False):
	"""Makes a new rainfall / waterdepth mono model.

	Args:
		metadata (dict): A dictionary of metadata about the dataset to use to build the model with.
		shape_water_out (int[]): The width and height (in that order) that should dictate the output shape of the segmentation head. CURRENTLY NOT USED.
		feature_dim	(int, optiona): The size of the bottleneck. Defaults to 512.
		model_arch_enc (str, optional): The architecture code for the underlying (inverted) ConvNeXt model for the encoder. Defaults to "convnext_xtiny".
		model_arch_dec (str, optional): The architecture code for the underlying (inverted) ConvNeXt model for the decoder. Defaults to "convnext_i_xtiny".
		batch_size (int, optional): The batch size. Reduce to save memory. Defaults to 64.
		water_bins (int, optional): The number of classes that the water depth output oft he segmentation head should be binned into. Defaults to 2.
		heightmap_input (bool, option): Whether a heightmap is being passed as an input to the model or not. Required to ensure we know how many channels the model will be taking in (the heightmap takes u  an additional input channel). Default: false.
		learning_rate (float, optional): The (initial) learning rate. YOU DO NOT USUALLY NEED TO CHANGE THIS. For experimental purposes only. Defaults to None, which means it will be determined automatically.

	Returns:
		tf.keras.Model: The new model, freshly compiled for your convenience! :D
	"""
	rainfall_channels, rainfall_height, rainfall_width = metadata["rainfallradar"] # shape = [channels, height, weight]
	# BUG: We somehow *still* have the rainfall radar data transposed incorrectly! I have no idea how this happened. dataset_mono fixes it with (another) transpose
	
	if heightmap_input:
		rainfall_channels += 1
	
	print("RAINFALL channels", rainfall_channels, "width", rainfall_width, "height", rainfall_height, "HEIGHTMAP_INPUT", heightmap_input)
	
	
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
	
	print("DEBUG:model ENCODER output_shape", layer_next.shape)
	
	# BOTTLENECK
	layer_next = tf.keras.layers.Dense(name="cns.stage.bottleneck.dense2", units=feature_dim)(layer_next)
	layer_next = tf.keras.layers.Activation(name="cns.stage.bottleneck.gelu2", activation="gelu")(layer_next)
	layer_next = tf.keras.layers.LayerNormalization(name="cns.stage.bottleneck.norm2", epsilon=1e-6)(layer_next)
	layer_next = tf.keras.layers.Dropout(name="cns.stage.bottleneck.dropout", rate=0.1)(layer_next)
	
	# DECODER
	layer_next = LayerStack2Image(target_width=4, target_height=4)(layer_next)
	# layer_next = tf.keras.layers.Reshape((4, 4, math.floor(feature_dim_in/(4*4))), name="cns.stable_begin.reshape")(layer_next)
	
	print("DEBUG:model BOTTLENECK:stack2image output_shape", layer_next.shape)
	
	layer_next = tf.keras.layers.Dense(name="cns.stage.begin.dense2", units=feature_dim)(layer_next)
	layer_next = tf.keras.layers.Activation(name="cns.stage_begin.relu2", activation="gelu")(layer_next)
	layer_next = tf.keras.layers.LayerNormalization(name="cns.stage_begin.norm2", epsilon=1e-6)(layer_next)
	
	layer_next = do_convnext_inverse(layer_next, arch_name=model_arch_dec)
	
	# TODO: An attention layer here instead of a dense layer, with a skip connection perhaps?
	logger.warning("Warning: TODO implement attention from https://ieeexplore.ieee.org/document/9076883")
	layer_next = tf.keras.layers.Dense(32, activation="gelu")(layer_next)
	# LOSS cross entropy
	# layer_next = tf.keras.layers.Conv2D(water_bins, activation="gelu", kernel_size=1, padding="same")(layer_next)
	# layer_next = tf.keras.layers.Softmax(axis=-1)(layer_next)
	# LOSS dice
	layer_next = tf.keras.layers.Conv2D(1, activation="gelu", kernel_size=1, padding="same")(layer_next)
	
	model = tf.keras.Model(
		inputs = layer_input,
		outputs = layer_next
	)
	
	logger.info(f"learning_rate: {str(learning_rate)}")
	
	optimizer = "Adam"
	if learning_rate is not None:
		optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
	model.compile(
		optimizer=optimizer,
		loss=LossCrossentropy(batch_size=batch_size),
		# loss=tf.keras.losses.CategoricalCrossentropy(),
		metrics=[tf.keras.metrics.CategoricalAccuracy()]
	)
	
	return model
