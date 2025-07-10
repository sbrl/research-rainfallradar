import tensorflow as tf


def read_weights(filepath_weights: str):
	with io.open(filepath_weights, "r", encoding="utf-8") as handle:
		weights = (
			handle.read().rstrip().split("\n")[1:]
		)  # split into rows & remove header

	print("DEBUG:weights", weights)

	[lower, upper, weight] = tf.io.decode_csv(
		weights,
		[
			# tf.constant(0, dtype=tf.int32), # row id - we don't care abt this col so it's excluded
			tf.constant(
				0, dtype=tf.float32
			),  # lower - Tensorflow is dumb so we hafta convert this later
			tf.constant(
				0, dtype=tf.float32
			),  # upper - Tensorflow is dumb so we hafta convert this later
			tf.constant(0, dtype=tf.float32),  # weight - full precision required here
		],
		field_delim="\t",
		select_cols=[1, 2, 3],
	)  # skip col 0

	# We hafta cast afterwards bc TF is dumb
	lower = tf.cast(lower, tf.float16)
	upper = tf.cast(upper, tf.float16)
	# weights are still tf.float32

	return lower, upper, weight


@tf.function
def find_sample_weight(value, col_lower, col_upper, col_weights):
	"""Finds the bin and hence weight associated with the given input (calculated total) value.
	
	All weights are precomputed sample weightings in the range 0..1.
	
	This is a Tensorflow Function.
	
	Args:
		value (tf.Tensor): The input predicted value from e.g. tf.math.reduce_sum or something
		col_lower (tf.Tensor): The col_lower column from read_weights.
		col_upper (tf.Tensor): The col_upper column from read_weights.
		col_weights (tf.Tensor): The col_weights column from read_weights.

	Returns:
		tf.Tensor: The associated sample weight value in the range 0..1
	"""
	# Note: broadcasting improves performance/efficiency here
	# Ref https://numpy.org/doc/stable/user/basics.broadcasting.html
	select_lower = tf.math.greater_equal(value, col_lower)
	select_upper = tf.math.less_equal(value, col_upper)

	# Identify the indices of the weighting table to preserve
	selected = tf.cast(
		tf.math.logical_and(select_lower, select_upper),
		dtype=tf.float32,  # we're abt to hit up the weights tbl, which is in tf.float32
	)

	# Extract the highest weight from the weighting table. This is required because if the input value falls EXACTLY on a bin boundary, then we may select multiple bins using this method
	return tf.math.reduce_max(selected * col_weights)
