import math

import tensorflow as tf

def sensitivity(y_true, y_pred):
	y_true = tf.cast(y_true, dtype=tf.float32)
	y_pred = tf.cast(y_pred, dtype=tf.float32)
	
	y_pred = tf.math.argmax(y_pred, axis=-1)
	
	recall = tf.keras.metrics.Recall()
	recall.update_state(y_true, y_pred)
	return recall.result()

class MetricSensitivity(tf.keras.metrics.Metric):
	"""An implementation of the sensitivity.
	Also known as Recall. In other words, how many of the true positives were accurately predicted.
	@source 
	Args:
		smooth (float): The batch size (currently unused).
	"""

	def __init__(self, name="sensitivity", **kwargs):
		super(MetricSensitivity, self).__init__(name=name)
		
		self.recall = tf.keras.metrics.Recall(**kwargs)

	def call(self, y_true, y_pred):
		ground_truth = tf.cast(y_true, dtype=tf.float32)
		prediction = tf.cast(y_pred, dtype=tf.float32)
		
		return self.recall(y_true, y_pred)

	def get_config(self):
		config = super(MetricSensitivity, self).get_config()
		config.update({
			
		})
		return config
