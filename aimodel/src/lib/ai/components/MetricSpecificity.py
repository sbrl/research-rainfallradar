import math

import tensorflow as tf


def specificity(y_pred, y_true):
	"""
	@source https://datascience.stackexchange.com/a/40746/86851
	param:
	y_pred - Predicted labels
	y_true - True labels 
	Returns:
	Specificity score
	"""
	neg_y_true = 1 - y_true
	neg_y_pred = 1 - y_pred
	fp = K.sum(neg_y_true * y_pred)
	tn = K.sum(neg_y_true * neg_y_pred)
	specificity = tn / (tn + fp + K.epsilon())
	return specificity


class MetricSpecificity(tf.keras.metrics.Metric):
	"""An implementation of the specificity.
	In other words, a measure of how many of the true negatives were accurately predicted
	@source 
	Args:
		smooth (float): The batch size (currently unused).
	"""

	def __init__(self, name="specificity", **kwargs):
		super(MetricSpecificity, self).__init__(name=name, **kwargs)

	def call(self, y_true, y_pred):
		ground_truth = tf.cast(y_true, dtype=tf.float32)
		prediction = tf.cast(y_pred, dtype=tf.float32)

		return specificity(ground_truth, prediction)

	def get_config(self):
		config = super(MetricSpecificity, self).get_config()
		config.update({
			
		})
		return config
