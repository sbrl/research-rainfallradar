import math

import tensorflow as tf

def dice_coef(y_true, y_pred, smooth=100):        
	"""Calculates the Dice coefficient.
	@source https://stackoverflow.com/a/72264322/1460422
	Args:
		y_true (Tensor): The ground truth.
		y_pred (Tensor): The predicted output.
		smooth (float, optional): The smoothness of the output. Lower values = penalise the model more for mistakes to make it better at fine detail. Defaults to 100.
	Returns:
		Tensor: The dice coefficient.
	"""
	# K.flatten → tf.reshape; K.sum → tf.math.reduce_sum
	y_true_f = tf.reshape(y_true, [-1])
	y_pred_f = tf.reshape(y_pred, [-1])
	intersection = tf.math.reduce_sum(y_true_f * y_pred_f)
	dice = (2. * intersection + smooth) / (tf.math.reduce_sum(y_true_f) + tf.math.reduce_sum(y_pred_f) + smooth)
	return dice

def dice_coef_loss(y_true, y_pred, **kwargs):
	"""Turns the dice coefficient into a loss value.
	NOTE: This is not the only option here. See also the other options in the source.
	@source https://stackoverflow.com/a/72264322/1460422
	Args:
		y_true (Tensor): The ground truth
		y_pred (Tensor): The predicted output.
	Returns:
		Tensor: The Dice coefficient, but as a loss value that decreases instead fo increases as the model learns.
	"""
	return -dice_coef(y_true, y_pred, **kwargs)



class LossDice(tf.keras.losses.Loss):
	"""An implementation of the dice loss function.
	Args:
		smooth (float): The batch size (currently unused).
	"""
	def __init__(self, smooth=100, **kwargs):
		super(LossDice, self).__init__(**kwargs)
		
		self.param_smooth = smooth
	
	def call(self, y_true, y_pred):
		return dice_coef_loss(y_true, y_pred, smooth=self.param_smooth)
	
	def get_config(self):
		config = super(LossDice, self).get_config()
		config.update({
			"smooth": self.param_smooth,
		})
		return config
