import math

from loguru import logger
import tensorflow as tf

from .components.convnext_inverse import do_convnext_inverse

def model_rainfallwater_segmentation(metadata, feature_dim_in, shape_water_out, batch_size=64, summary_file=None):
	
	layer_input = tf.keras.layers.Input(
		shape=(feature_dim_in)
	)
	
	# BEGIN
	layer_next = tf.keras.layers.Dense(name="cns.stage.begin.dense")(layer_input)
	layer_next = tf.keras.layers.LayerNormalisation(name="stage_begin.norm", epsilon=1e-6)(layer_next)
	layer_next = tf.keras.layers.ReLU(name="stage_begin.relu")(layer_next)
	
	layer_next = do_convnext_inverse(layer_next, arch_name="convnext_i_tiny")
	
	# TODO: Implement projection head here
	
	model = tf.keras.Model(
		inputs = layer_input,
		outputs = layer_next
	)
	
	model.compile(
		optimizer="Adam",
		loss="" # TODO: set this to binary cross-entropy loss
	)
	
	return model