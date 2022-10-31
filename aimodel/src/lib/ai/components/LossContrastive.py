import math

import tensorflow as tf

class LossContrastive(tf.keras.losses.Loss):
	def __init__(self, weight_temperature, batch_size):
		super(LossContrastive, self).__init__()
		self.batch_size = batch_size
		self.weight_temperature = weight_temperature
	
	def call(self, y_true, y_pred):
		rainfall, water = tf.unstack(y_pred, axis=-2)
		# print("LOSS:call y_true", y_true.shape)
		# print("LOSS:call y_pred", y_pred.shape)
		print("START rainfall", rainfall)
		print("START water", water)
		
		# # Ensure the shapes are defined
		# rainfall = tf.reshape(rainfall, [self.batch_size, rainfall.shape[1]])
		# water = tf.reshape(water, [self.batch_size, water.shape[1]])
		
		# normalise features
		# rainfall = rainfall / tf.math.l2_normalize(rainfall, axis=1)
		# water = water / tf.math.l2_normalize(water, axis=1)
		
		print("AFTER_L2 rainfall", rainfall)
		print("AFTER_L2 water", water)
		
		# logits = tf.linalg.matmul(rainfall, tf.transpose(water)) * tf.clip_by_value(tf.math.exp(self.weight_temperature), 0, 100)
		logits = tf.linalg.matmul(rainfall, tf.transpose(water)) * tf.math.exp(self.weight_temperature)
		
		print("LOGITS", logits)
		
		# labels			= tf.eye(self.batch_size, dtype=tf.int32) # we *would* do this if we were using mean squared error...
		labels			= tf.range(self.batch_size, dtype=tf.int32) # each row is a different category we think
		loss_rainfall	= tf.keras.metrics.sparse_categorical_crossentropy(labels, logits, from_logits=True, axis=0)
		loss_water		= tf.keras.metrics.sparse_categorical_crossentropy(labels, logits, from_logits=True, axis=1)
		
		
		# loss_rainfall	= tf.keras.metrics.binary_crossentropy(labels, logits, from_logits=True, axis=0)
		# loss_water		= tf.keras.metrics.binary_crossentropy(labels, logits, from_logits=True, axis=1)
		print("LABELS", labels)
		print("LOSS_RAINFALL", loss_rainfall)
		print("LOSS_WATER", loss_water)
				
		loss = (loss_rainfall + loss_water) / 2
		
		print("LOSS", loss)
		
		loss = tf.math.reduce_mean(loss)
		
		print("LOSS FINAL", loss)
		# cosine_similarity results in tensor of range -1 - 1, but tf.sparse.eye has range 0 - 1
		# print("LABELS", labels)
		# print("LOSS_rainfall", loss_rainfall)
		# print("LOSS_water", loss_water)
		# print("LOSS", loss)
		return loss


if __name__ == "__main__":
	weight_temperature = tf.Variable(name="loss_temperature", shape=1, initial_value=tf.constant([
		math.log(1 / 0.07)
	]))
	loss = LossContrastive(weight_temperature=weight_temperature, batch_size=64)
	
	tensor_input = tf.random.uniform([64, 2, 512])
	print(loss(tf.constant(1), tensor_input))