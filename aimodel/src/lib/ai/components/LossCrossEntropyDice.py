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
	The ground truth labels should sparse, NOT one-hot. The predictions should be one-hot, NOT sparse.
	@source https://lars76.github.io/2018/09/27/loss-functions-for-segmentation.html#9
	log_cosh (bool): Whether to do log(cosh(dice_loss)) instead of just dice_loss on its own. Ref https://doi.org/10.1109/cibcb48159.2020.9277638
	"""
	
	def __init__(self, log_cosh=True, **kwargs):
		super(LossCrossEntropyDice, self).__init__(**kwargs)
		
		self.param_log_cosh = log_cosh
	
	def call(self, y_true, y_pred):
		y_true = tf.cast(y_true, tf.float32)
		y_true = tf.one_hot(tf.cast(y_true, dtype=tf.int32), 2)  # Input is sparse
		
		cel = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred)
		dice = dice_loss(y_true, y_pred)
		
		o = cel + dice
		return tf.reduce_mean(o)
	
	def get_config(self):
		config = super(LossCrossEntropyDice, self).get_config()
		config.update({
			"log_cosh": self.param_log_cosh
		})
		return config
