import tensorflow as tf


def make_one_hot_mean_iou(classes=2):
	iou = tf.keras.metrics.MeanIoU(num_classes=classes)
	def one_hot_mean_iou(y_true, y_pred, ):
		"""Compute the mean IoU for one-hot tensors.
		Args:
			y_true (tf.Tensor): The ground truth label.
			y_pred (tf.Tensor): The output predicted by the model.
		
		Returns:
			tf.Tensor: The computed mean IoU.
		"""
		print("DEBUG:meaniou y_pred.shape BEFORE", y_pred.shape)
		print("DEBUG:meaniou y_true.shape BEFORE", y_true.shape)
		y_pred = tf.math.argmax(y_pred, axis=-1)
		y_true = tf.cast(y_true, dtype=tf.float32)
		y_pred = tf.cast(y_pred, dtype=tf.float32)
		print("DEBUG:meaniou y_pred.shape AFTER", y_pred.shape)
		print("DEBUG:meaniou y_true.shape AFTER", y_true.shape)
		
		iou.reset_state()
		iou.update_state(y_true, y_pred)
		return iou.result()
	return one_hot_mean_iou