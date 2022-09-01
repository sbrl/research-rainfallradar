
from loguru import logger
import tensorflow as tf

from .components.LayerContrastiveEncoder import LayerContrastiveEncoder
from .components.LayerCheeseMultipleOut import LayerCheeseMultipleOut
from .components.LossContrastive import LossContrastive

def model_rainfallwater_contrastive(shape_rainfall, shape_water, batch_size=64, feature_dim=2048):
	logger.info(shape_rainfall)
	logger.info(shape_water)
	
	# Shapes come from what rainfallwrangler sees them as, but we add an extra dimension when reading the .tfrecord file
	rainfall_width, rainfall_height, rainfall_channels = shape_rainfall # shape = [width, height, channels]
	water_width, water_height = shape_water # shape = [width, height]
	water_channels = 1 # added in dataset → make_dataset → parse_item
	
	input_rainfall = tf.keras.layers.Input(
		shape=shape_rainfall
	)
	input_water = tf.keras.layers.Input(
		shape=(water_width, water_height, water_channels)
	)
	
	
	rainfall = LayerContrastiveEncoder(
		input_width=rainfall_width,
		input_height=rainfall_height,
		channels=rainfall_channels,
		feature_dim=feature_dim
	)(input_rainfall)
	water = LayerContrastiveEncoder(
		input_width=water_width,
		input_height=water_height,
		channels=water_channels,
		feature_dim=feature_dim
	)(input_water)
	
	
	layer_final = LayerCheeseMultipleOut()
	final = layer_final([ rainfall, water ])
	weight_temperature = layer_final.weight_temperature
	
	model = tf.keras.Model(
		inputs = [ input_rainfall, input_water ],
		outputs = final
	)
	
	model.compile(
		optimizer="Adam",
		loss=LossContrastive(batch_size=batch_size, weight_temperature=weight_temperature)
	)
	
	return model