import math

from curses import meta
from loguru import logger
import tensorflow as tf

from .components.LayerContrastiveEncoder import LayerContrastiveEncoder
from .components.LayerCheeseMultipleOut import LayerCheeseMultipleOut
from .components.LossContrastive import LossContrastive

def model_rainfallwater_contrastive(metadata, shape_water, batch_size=64, feature_dim=2048, summary_file=None):
	# Shapes come from what rainfallwrangler sees them as, but we add an extra dimension when reading the .tfrecord file
	rainfall_channels, rainfall_width, rainfall_height = metadata["rainfallradar"] # shape = [channels, width, height]
	water_width, water_height = shape_water # shape = [width, height]
	water_channels = 1 # added in dataset → make_dataset → parse_item
	
	rainfall_width, rainfall_height = math.floor(rainfall_width / 2), math.floor(rainfall_height / 2)
	
	logger.info("SOURCE shape_rainfall " + str(metadata["rainfallradar"]))
	logger.info("SOURCE shape_water " + str(metadata["waterdepth"]))
	logger.info("TARGET shape_water" + str(shape_water))
	logger.info("TARGET shape_rainfall" + str([ rainfall_width, rainfall_height, rainfall_channels ]))
	
	
	input_rainfall = tf.keras.layers.Input(
		shape=(rainfall_width, rainfall_height, rainfall_channels)
	)
	input_water = tf.keras.layers.Input(
		shape=(water_width, water_height, water_channels)
	)
	
	print("MAKE ENCODER rainfall")
	rainfall = LayerContrastiveEncoder(
		input_width=rainfall_width,
		input_height=rainfall_height,
		channels=rainfall_channels,
		feature_dim=feature_dim,
		summary_file=summary_file,
		arch_name="convnext_tiny",
	)(input_rainfall)
	print("MAKE ENCODER water")
	water = LayerContrastiveEncoder(
		input_width=water_width,
		input_height=water_height,
		channels=water_channels,
		feature_dim=feature_dim,
		arch_name="convnext_xtiny",
		summary_file=summary_file
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