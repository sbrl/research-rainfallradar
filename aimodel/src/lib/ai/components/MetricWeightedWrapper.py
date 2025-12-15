import tensorflow as tf

from .WeightedEngine import find_sample_weight

def make_weighted_metric(fn: callable, tbl_weights) -> callable:
		
	def inner_metric(y_true, y_pred):
		value = fn(y_true, y_pred)
		value = tf.math.reduce_sum(value)
		
		find_sample_weight(value, *tbl_weights)
	
	return inner_metric