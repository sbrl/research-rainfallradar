import math

import tensorflow as tf


class LossCrossentropy(tf.keras.losses.Loss):
	"""Wraps the cross-entropy loss function because it's buggy.
	@warning: tf.keras.losses.CategoricalCrossentropy() isn't functioning as intended during training...
	Args:
		batch_size (integer): The batch size (currently unused).
	"""
	def __init__(self, batch_size):
		super(LossCrossentropy, self).__init__()
		
		self.param_batch_size = batch_size
	
	def call(self, y_true, y_pred):
		result = tf.keras.metrics.categorical_crossentropy(y_true, y_pred)
		result_reduce = tf.math.reduce_sum(result)
		label_nowater = tf.math.reduce_sum(tf.argmax(y_true, axis=-1))
		tf.print("DEBUG:TFPRINT:loss LABEL", y_true.shape, y_true, "PREDICT", y_pred.shape, y_pred, "BEFORE_REDUCE", result.shape, result, "AFTER_REDUCE", result_reduce.shape, result_reduce)
		return result
	
	def get_config(self):
		config = super(LossCrossentropy, self).get_config()
		config.update({
			"batch_size": self.param_batch_size,
		})
		return config


if __name__ == "__main__":
	weight_temperature = tf.Variable(name="loss_temperature", shape=1, initial_value=tf.constant([
		math.log(1 / 0.07)
	]))
	loss = LossCrossentropy(weight_temperature=weight_temperature, batch_size=64)
	
	tensor_input = tf.random.uniform([64, 2, 512])
	print(loss(tf.constant(1), tensor_input))