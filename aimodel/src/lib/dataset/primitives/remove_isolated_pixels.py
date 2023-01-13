import tensorflow as tf


def remove_isolated_pixels(binarised_water_labels):
	# we expect the data in the form [ height, width ], where each value is either 1 or 0 (i.e. BEFORE any one-hot)
	
	data = tf.expand_dims(tf.expand_dims(binarised_water_labels, axis=0), axis=-1)
	
	conv = tf.squeeze(tf.nn.conv2d(data, tf.ones([3,3,1,1]), 1, "SAME"))
	
	data_map_remove = tf.cast(tf.math.equal(tf.math.multiply(
		binarised_water_labels,
		conv
	), 1), tf.float32)
	
	return tf.math.subtract(binarised_water_labels, data_map_remove)
