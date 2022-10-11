import math

from loguru import logger
import tensorflow as tf

from .components.convnext_inverse import do_convnext_inverse

def model_rainfallwater_segmentation(metadata, feature_dim_in, shape_water_out, model_arch="convnext_i_xtiny", batch_size=64, summary_file=None):
	out_water_width, out_water_height = shape_water_out
	
	
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
	
	layer_next = do_convnext_inverse(layer_next, arch_name="convnext_i_tiny")
	
	# TODO: An attention layer here instead of a dense layer, with a skip connection perhaps?
	logger.warning("Warning: TODO implement attention from https://ieeexplore.ieee.org/document/9076883")
	layer_next = tf.keras.layers.Dense(32)(layer_next)
	layer_next = tf.keras.layers.Conv2D(1, kernel_size=1, activation="softmax", padding="same")(layer_next)
	
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