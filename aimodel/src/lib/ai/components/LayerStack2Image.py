import tensorflow as tf

# Code from https://github.com/leanderme/ConvNeXt-Tensorflow/blob/main/ConvNeXt.ipynb

class LayerStack2Image(tf.keras.layers.Layer):
	def __init__(self, target_width, target_height, name=None, **kwargs):
		super(LayerStack2Image, self).__init__(name=name)
		
		self.param_target_width = target_width
		self.param_target_height = target_height
	
	def get_config(self):
		config = super(LayerStack2Image, self).get_config()
		config.update({
			"target_width": self.param_target_width,
			"target_height": self.param_target_height,
		})
		
		return config
	
	def call(self, input_thing, **kwargs):
		result = tf.stack([ input_thing for i in range(self.param_target_width) ], axis=-1)
		result = tf.stack([ result for i in range(self.param_target_height) ], axis=-1)
		result = tf.stack([ result ], axis=-1) # channel dimension
		return result
	