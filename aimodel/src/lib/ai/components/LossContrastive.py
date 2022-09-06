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
		# print("BEFORE_RESHAPE rainfall", rainfall)
		# print("BEFORE_RESHAPE water", water)
		
		# # Ensure the shapes are defined
		# rainfall = tf.reshape(rainfall, [self.batch_size, rainfall.shape[1]])
		# water = tf.reshape(water, [self.batch_size, water.shape[1]])
		
		
		logits = tf.linalg.matmul(rainfall, tf.transpose(water)) * tf.clip_by_value(tf.math.exp(self.weight_temperature), 0, 100)
		
		# print("LOGITS", logits)
		
		labels			= tf.eye(self.batch_size, dtype=tf.int32)
		loss_rainfall	= tf.keras.metrics.binary_crossentropy(labels, logits, from_logits=True, axis=0)
		loss_water		= tf.keras.metrics.binary_crossentropy(labels, logits, from_logits=True, axis=1)
		
		loss = (loss_rainfall + loss_water) / 2
		
		# cosine_similarity results in tensor of range -1 - 1, but tf.sparse.eye has range 0 - 1
		# print("LABELS", labels)
		# print("LOSS_rainfall", loss_rainfall)
		# print("LOSS_water", loss_water)
		# print("LOSS", loss)
		return loss
	