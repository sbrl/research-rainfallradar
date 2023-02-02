import tensorflow as tf

# Code from https://github.com/leanderme/ConvNeXt-Tensorflow/blob/main/ConvNeXt.ipynb

class LayerConvNeXtGamma(tf.keras.layers.Layer):
	def __init__(self, const_val = 1e-6, dim = None, name=None, **kwargs):
		super(LayerConvNeXtGamma, self).__init__(name=name)
		
		self.dim   = dim
		self.const = const_val * tf.ones((self.dim), dtype=tf.float32 if tf.mixed_precision.global_policy().name == "float32" else tf.float16)

	def call(self, inputs, **kwargs):
		return tf.multiply(inputs, self.const)
	
	def get_config(self):
		config = super(LayerConvNeXtGamma, self).get_config()
		
		config.update({ "const_val": self.const.numpy(), "dim": self.dim })
		return config
