import math

import tensorflow as tf


def one_hot_mean_iou(y_true, y_pred, classes=2):
	"""Compute the mean IoU for one-hot tensors.
	Args:
		y_true (tf.Tensor): The ground truth label.
		y_pred (tf.Tensor): The output predicted by the model.
	
	Returns:
		tf.Tensor: The computed mean IoU.
	"""
	
	y_pred = tf.math.argmax(y_pred, axis=-1)
	y_true = tf.cast(y_true, dtype=tf.float32)
	y_pred = tf.cast(y_pred, dtype=tf.float32)
	
	
	iou = tf.keras.metrics.MeanIoU(num_classes=classes)
	iou.update_state(y_true, y_pred)
	return iou.result()
