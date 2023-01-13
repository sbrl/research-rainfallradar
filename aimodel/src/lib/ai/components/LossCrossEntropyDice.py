import math

import tensorflow as tf


def dice_loss(y_true, y_pred):
	"""Compute Dice loss.
	@source https://lars76.github.io/2018/09/27/loss-functions-for-segmentation.html#9
	Args:
		y_true (tf.Tensor): The ground truth label.
		y_pred (tf.Tensor): The output predicted by the model.
	
	Returns:
		tf.Tensor: The computed Dice loss.
	"""
	y_pred = tf.math.sigmoid(y_pred)
	numerator = 2 * tf.reduce_sum(y_true * y_pred)
	denominator = tf.reduce_sum(y_true + y_pred)

	return 1 - numerator / denominator

class LossCrossEntropyDice(tf.keras.losses.Loss):
	"""Cross-entropy loss and dice loss combined together into one nice neat package.
	Combines the two with mean.
	@source https://lars76.github.io/2018/09/27/loss-functions-for-segmentation.html#9
	"""

	def __init__(self, **kwargs):
		super(LossDice, self).__init__(**kwargs)

	def call(self, y_true, y_pred):
		y_true = tf.cast(y_true, tf.float32)
		o = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred) + dice_loss(y_true, y_pred)
		return tf.reduce_mean(o)

	def get_config(self):
		config = super(LossDice, self).get_config()
		config.update({
			
		})
		return config
