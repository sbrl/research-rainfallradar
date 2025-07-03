#!/usr/bin/env python3
import os
import io
import tensorflow as tf

from LossCrossEntropyDice import LossCrossEntropyDice


def read_weights(filepath_weights: str):
	with io.open(filepath_weights, "r", encoding="utf-8") as handle:
		weights = handle.read().rstrip().split("\n")[1:] # split into rows & remove header
	
	print("DEBUG:weights", weights)
	
	[ lower, upper, weight ] = tf.io.decode_csv(weights, [
		# tf.constant(0, dtype=tf.int32), # row id - we don't care abt this col so it's excluded
		tf.constant(0, dtype=tf.float32), # lower - Tensorflow is dumb so we hafta convert this later
		tf.constant(0, dtype=tf.float32), # upper - Tensorflow is dumb so we hafta convert this later
		tf.constant(0, dtype=tf.float32), # weight - full precision required here
	], field_delim="\t", select_cols=[1,2,3]) # skip col 0
	
	# We hafta cast afterwards bc TF is dumb
	lower = tf.cast(lower, tf.float16)
	upper = tf.cast(upper, tf.float16)
	# weights are still tf.float32
	
	return lower, upper, weight
	
	

class LossWeightedCrossEntropyDice(LossCrossEntropyDice):
    """A weighted version of LossCrossEntropyDice. Takes a PRECOMPUTED weights file and uses that to weight each sample as it comes through by binning it via summation of ground-truth weights.

    In other words, this is a focal loss variant of the aforementioned class.

    Inherits from LossCrossEntropyDice.
    """

    def __init__(
        self,
        filepath_weights: str | None = None,
        col_lower: tf.Tensor | None = None,
        col_upper: tf.Tensor | None = None,
        col_weights: tf.Tensor | None = None,
        **kwargs,
    ):
        super(LossCrossEntropyDice, **kwargs)

        # When config is passed back in it is done by passing to the constructor - hence the loading here
        if col_lower is not None and col_upper is not None and col_weights is not None:
            self.col_lower = col_lower
            self.col_upper = col_upper
            self.col_weights = col_upper
        elif type(filepath_weights) is str:
            self.col_lower, self.col_upper, self.col_weights = read_weights(
                filepath_weights
            )
        else:
            raise Exception(
                "Error: both fileapth_weights and (col_lower || col_upper || col_weights)  were None"
            )

    def call(self, y_true, y_pred, **kwargs):
        label_rainfall = tf.cast(y_true, dtype=tf.float16)

        label_rainfall_total = tf.cast(
            tf.math.reduce_sum(label_rainfall), dtype=tf.float16
        )  # max range -65K - +65K, but that should be Fine™ bc our max range is only a couple a K

        val_loss = super(LossWeightedCrossEntropyDice, self).call(
            y_true, y_pred, **kwargs
        )

        # Note: broadcasting improves performance/efficiency here
        # Ref https://numpy.org/doc/stable/user/basics.broadcasting.html
        select_lower = tf.math.greater_equal(label_rainfall_total, self.col_lower)
        select_upper = tf.math.less_equal(label_rainfall_total, self.col_upper)

        # Identify the indices of the weighting table to preserve
        selected = tf.cast(
            tf.math.logical_and(select_lower, select_upper),
            dtype=tf.float32,  # we're abt to hit up the weights tbl, which is in tf.float32
        )

        # Extract the highest weight from the weighting table. This is required because if the input value falls EXACTLY on a bin boundary, then we may select multiple bins using this method
        selected_weight = tf.math.reduce_max(selected * self.col_weights)

        return selected_weight * val_loss

        # finish filling this in - we have everything we need..... probably :P  (done)
        # gl to future me who will be implementing all the nasty tensor manipulation code here since you can't drop to normal Python/numpy data types bc of execution graphs :P
        # thanks past me, it was real fun --future me

    def get_config(self):
        config = super(LossWeightedCrossEntropyDice, self).get_config()
        config.update(
            {
                "col_lower": self.col_lower,
                "col_upper": self.col_upper,
                "col_weights": self.col_weights,
            }
        )
		

if __name__ == "__main__":
	col_lower, col_upper, col_weights = read_weights(os.environ["FILEPATH_WEIGHTS"])
	print("DEBUG:tensor_weights", col_lower, col_upper, col_weights)
