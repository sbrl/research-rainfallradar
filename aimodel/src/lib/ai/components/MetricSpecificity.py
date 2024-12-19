import tensorflow as tf


def specificity(y_true, y_pred):
	"""An implementation of the specificity.
	In other words, a measure of how many of the true negatives were accurately predicted
	@source https://datascience.stackexchange.com/a/40746/86851
	param:
	y_pred - Predicted labels
	y_true - True labels 
	Returns:
	Specificity score
	"""
	print("DEBUG:specificity y_pred.shape BEFORE", y_pred.shape)
	print("DEBUG:specificity y_true.shape BEFORE", y_true.shape)
	y_pred = tf.math.argmax(y_pred, axis=-1)
	y_true = tf.cast(y_true, dtype=tf.float32)
	y_pred = tf.cast(y_pred, dtype=tf.float32)
	print("DEBUG:specificity y_pred.shape AFTER", y_pred.shape)
	print("DEBUG:specificity y_true.shape AFTER", y_true.shape)
	
	neg_y_true = 1 - y_true
	neg_y_pred = 1 - y_pred
	fp = tf.math.reduce_sum(neg_y_true * y_pred)
	tn = tf.math.reduce_sum(neg_y_true * neg_y_pred)
	specificity = tn / (tn + fp + tf.keras.backend.epsilon())
	return specificity
