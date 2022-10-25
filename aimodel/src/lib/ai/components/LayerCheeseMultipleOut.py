import math
import tensorflow as tf


class LayerCheeseMultipleOut(tf.keras.layers.Layer):
	
	def __init__(self, batch_size, feature_dim, **kwargs):
		"""Creates a new cheese multiple out layer.
		This layer is useful if you have multiple outputs and a custom loss function that requires multiple inputs.
		Basically, it just concatenates all inputs.
		Inputs are expected to be in the form [ batch_size, feature_dim ], and this layer outputs in the form [ batch_size, concat, feature_dim ].
		This layer also creates a temperature weight for contrastive learning models.
		"""
		super(LayerCheeseMultipleOut, self).__init__(**kwargs)
		
		self.param_batch_size = batch_size
		self.param_feature_dim = feature_dim
		
		self.weight_temperature = tf.Variable(name="loss_temperature", shape=1, initial_value=tf.constant([0.07]))
		self.weight_nce = tf.Variable(
			name="loss_nce",
			shape=(batch_size, feature_dim),
			initial_value=tf.random.truncated_normal(
				(feature_dim),
				stddev=1.0 / math.sqrt(128)
			)
		)
		self.weight_nce_bias = tf.Variable(
			name="loss_nce_bias",
			shape=(feature_dim),
			initial_value=tf.zeros((feature_dim))
		)
	
	def get_config(self):
		config = super(LayerCheeseMultipleOut, self).get_config()
		config["batch_size"] = self.param_batch_size
		config["feature_dim"] = self.param_feature_dim
		return config
	
	def call(self, inputs):
		# inputs form: [ rainfall, water ]
		
		# By this point, the above has already dropped through the encoder, so should be in the form [ batch_size, dim ]
		
		return tf.stack(inputs, axis=-2)