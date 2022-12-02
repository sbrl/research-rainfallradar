import tensorflow as tf



class LayerCBAMAttentionSpatial(tf.keras.layers.Layer):
	def __init__(self, dim, **kwargs):
		super(LayerCBAMAttentionSpatial, self).__init__(**kwargs)

		self.param_dim = dim
		
		self.conv2d = tf.keras.layers.Conv2D(self.param_dim, kernel_size=7, padding="same", activation="sigmoid")

	def get_config(self):
		config = super(LayerCBAMAttentionSpatial, self).get_config()
		config.update({
			"dim": self.param_dim
		})
		return config

	def call(self, input_thing, training, **kwargs):
		
		pooled_max = tf.math.argmax(input_thing, axis=-1)
		pooled_avg = tf.math.reduce_mean(input_thing, axis=-1)
		
		result = tf.stack([pooled_max, pooled_avg])
		result = self.conv2d(result)
		
		return result


class LayerCBAMAttentionChannel(tf.keras.layers.Layer):
	def __init__(self, dim, reduction_ratio=1, **kwargs):
		super(LayerCBAMAttentionSpatial, self).__init__(**kwargs)

		self.param_dim = dim
		self.param_reduction_ratio = reduction_ratio
		
		self.mlp = tf.keras.Sequential([
			tf.keras.layers.Dense(self.param_dim / self.param_reduction_ratio),
			tf.keras.layers.Dense(self.param_dim)
		])

	def get_config(self):
		config = super(LayerCBAMAttentionSpatial, self).get_config()
		config.update({
			"dim": self.param_dim,
			"reduction_ratio": self.param_reduction_ratio
		})
		return config

	def call(self, input_thing, training, **kwargs):
		pooled_max = tf.nn.max_pool2d(input_thing, ksize=input_thing.shape[1:3])
		pooled_avg = tf.nn.avg_pool2d(input_thing, ksize=input_thing.shape[1:3])
		
		pooled_max = self.mlp(pooled_max)
		pooled_avg = self.mlp(pooled_avg)
		
		result = tf.math.sigmoid(pooled_max + pooled_avg)
		
		return result


def cbam_attention_spatial(input_thing, dim):
	pooled_max = tf.keras.layers.Lambda(lambda tensor: tf.math.argmax(tensor, axis=-1))(input_thing)
	pooled_avg = tf.keras.layers.Lambda(lambda tensor: tf.math.reduce_mean(tensor, axis=-1))
	
	pooled_max = tf.keras.layers.Dense(dim)(pooled_max)
	
	layer = tf.keras.layers.Concatenate()([pooled_max, pooled_avg])

cbam_id_next = 0

def cbam(input_thing, dim):
	"""Runs input_thing through CBAM.
	If you have a CNN-based model with skip connections, this layer would be placed at the end of a block directly BEFORE the skip connection rejoins.

	Args:
		input_thing (tf.Tensor): The input layer to operate on.
		dim (int): The size of the feature dimension.

	Returns:
		tf.Tensor: The input after being run through CBAM.
	"""
	
	id_this = cbam_id_next
	cbam_id_next += 1
	
	layer = input_thing
	
	attn_channel = LayerCBAMAttentionChannel(dim, name=f"cbam{id_this}.attn.channel")(input_thing)
	
	layer = tf.keras.layers.Multiply(name=f"cbam{id_this}.mult1")([layer, attn_channel])
	
	attn_spatial = LayerCBAMAttentionSpatial(dim, name=f"cbam{id_this}.attn.spatial")(input_thing)
	
	layer = tf.keras.layers.Multiply(name=f"cbam{id_this}.mult2")([layer, attn_spatial])
	
	return layer