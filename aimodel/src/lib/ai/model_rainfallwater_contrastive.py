
from loguru import logger
import tensorflow as tf

from .components.LayerContrastiveEncoder import LayerContrastiveEncoder
from .components.LayerCheeseMultipleOut import LayerCheeseMultipleOut
from .components.LossContrastive import LossContrastive

def model_rainfallwater_contrastive(shape_rainfall, shape_water, feature_dim=200):
	logger.info(shape_rainfall)
	logger.info(shape_water)
	rainfall_width, rainfall_height, rainfall_channels = shape_rainfall
	water_width, water_height, water_channels = shape_water
	
	input_rainfall = tf.keras.layers.Input(
		shape=shape_rainfall
	)
	input_water = tf.keras.layers.Input(
		shape=shape_water
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
	
	
	final = LayerCheeseMultipleOut()([ rainfall, water ])
	weight_temperature = final.weight_temperature
	
	model = tf.keras.Model(
		inputs = [ input_rainfall, input_water ],
		outputs = final
	)
	
	model.compile(
		optimizer="Adam",
		loss=LossContrastive(weights_temperature=weight_temperature)
	)