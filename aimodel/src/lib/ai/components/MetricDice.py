import math

import tensorflow as tf


def dice_coefficient(y_true, y_pred):
	"""Compute the dice coefficient.
	A measure of how similar 2 things are [images], or how much they overlap [image segmentation]
	@source https://lars76.github.io/2018/09/27/loss-functions-for-segmentation.html#9
	Args:
		y_true (tf.Tensor): The ground truth label.
		y_pred (tf.Tensor): The output predicted by the model.
	
	Returns:
		tf.Tensor: The computed Dice coefficient.
	"""
	y_pred = tf.math.sigmoid(y_pred)
	numerator = 2 * tf.reduce_sum(y_true * y_pred)
	denominator = tf.reduce_sum(y_true + y_pred)

	return numerator / denominator

class MetricDice(tf.keras.metrics.Metric):
	"""An implementation of the dice loss function.
	@source 
	Args:
		smooth (float): The batch size (currently unused).
	"""
	def __init__(self, name="dice_coefficient", smooth=100, **kwargs):
		super(MetricDice, self).__init__(name=name, **kwargs)
		
		self.param_smooth = smooth
	
	def call(self, y_true, y_pred):
		ground_truth = tf.cast(y_true, dtype=tf.float32)
		prediction = tf.cast(y_pred, dtype=tf.float32)
		
		return dice_coef(ground_truth, prediction, smooth=self.param_smooth)
	
	def get_config(self):
		config = super(MetricDice, self).get_config()
		config.update({
			"smooth": self.param_smooth,
		})
		return config
