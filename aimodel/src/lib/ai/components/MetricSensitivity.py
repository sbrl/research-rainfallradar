import math

import tensorflow as tf


def make_sensitivity():
	recall = tf.keras.metrics.Recall()
	def sensitivity(y_true, y_pred):
		print("DEBUG:sensitivity y_pred.shape BEFORE", y_pred.shape)
		print("DEBUG:sensitivity y_true.shape BEFORE", y_true.shape)
		y_pred = tf.math.argmax(y_pred, axis=-1)
		y_true = tf.cast(y_true, dtype=tf.float32)
		y_pred = tf.cast(y_pred, dtype=tf.float32)
		print("DEBUG:sensitivity y_pred.shape AFTER", y_pred.shape)
		print("DEBUG:sensitivity y_true.shape AFTER", y_true.shape)
		
		recall.reset_state()
		recall.update_state(y_true, y_pred)
		return recall.result()
	
	return sensitivity