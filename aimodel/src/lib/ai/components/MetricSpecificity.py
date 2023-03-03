import math

import tensorflow as tf


def specificity(y_pred, y_true):
	"""An implementation of the specificity.
	In other words, a measure of how many of the true negatives were accurately predicted
	@source https://datascience.stackexchange.com/a/40746/86851
	param:
	y_pred - Predicted labels
	y_true - True labels 
	Returns:
	Specificity score
	"""
	
	y_true = tf.cast(y_true, dtype=tf.float32)
	y_pred = tf.cast(y_pred, dtype=tf.float32)
	
	neg_y_true = 1 - y_true
	neg_y_pred = 1 - y_pred
	fp = K.sum(neg_y_true * y_pred)
	tn = K.sum(neg_y_true * neg_y_pred)
	specificity = tn / (tn + fp + K.epsilon())
	return specificity
