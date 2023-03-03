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
	
	y_true = tf.cast(y_true, dtype=tf.float32)
	y_pred = tf.cast(y_pred, dtype=tf.float32)
	
	y_pred = tf.math.sigmoid(y_pred)
	numerator = 2 * tf.reduce_sum(y_true * y_pred)
	denominator = tf.reduce_sum(y_true + y_pred)
	
	return numerator / denominator


def metric_dice_coefficient(y_true, y_pred):
	y_pred = tf.math.argmax(y_pred, axis=-1)
	return dice_coefficient(y_true, y_pred)