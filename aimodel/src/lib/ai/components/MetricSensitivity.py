import math

import tensorflow as tf

def sensitivity(y_true, y_pred):
	y_pred = tf.math.argmax(y_pred, axis=-1)
	y_true = tf.cast(y_true, dtype=tf.float32)
	y_pred = tf.cast(y_pred, dtype=tf.float32)
	
	recall = tf.keras.metrics.Recall()
	recall.update_state(y_true, y_pred)
	return recall.result()
