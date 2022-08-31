import tensorflow as tf

from tensorflow.keras.applications.resnet_v2 import ResNet50V2
# from transformers import TFConvNextModel, ConvNextConfig
from ..helpers.summarywriter import summarylogger

class LayerContrastiveEncoder(tf.keras.layers.Layer):
	
	def __init__(self, input_width, input_height, channels, feature_dim=200, **kwargs):
		"""Creates a new contrastive learning encoder layer.
		While this is intended for contrastive learning, this can (in theory) be used anywhere as it's just a generic wrapper layer.
		The key feature here is that it does not care about the input size or the number of channels.
		Currently it uses a ResNetV2 internally, but an upgrade to ConvNeXt is planned once Tensorflow Keras' implementation comes out of nightly and into stable.
		We would use ResNetRS (as it's technically superior), but the implementation is bad and in places outright *wrong* O.o
		
		Args:
			feature_dim (int, optional): The size of the features dimension in the output shape. Note that there are *two* feature dimensions outputted - one for the left, and one for the right. They will both be in the form [ batch_size, feature_dim ]. Set to a low value (e.g. 25) to be able to plot a sensible a parallel coordinates graph. Defaults to 200.
			image_width (int): The size of width of the input in pixels.
			image_height (int): The size of height of the input in pixels.
			channels (int): The number of channels in the input in pixels.
		"""
		super(LayerContrastiveEncoder, self).__init__(**kwargs)
		
		self.param_input_width	= input_width
		self.param_input_height	= input_height
		self.param_channels		= channels
		self.param_feature_dim	= feature_dim
		
		"""The main ResNet model that forms the encoder.
		Note that both the left AND the right go through the SAME encoder!s
		"""
		self.encoder = ResNet50V2(
			include_top=False,
			input_shape=(self.param_channels, self.param_input_width, self.param_input_height),
			weights=None,
			pooling=None,
			data_format="channels_first"
		)
		"""Small sequential stack of layers that control the size of the outputted feature dimension.
		"""
		self.embedding = tf.keras.layers.Dense(self.param_feature_dim)
		self.embedding_input_shape = [None, 2048] # The output shape of the above ResNet AFTER reshaping.
		
		summarylogger(self.encoder)
	
	def get_config(self):
		config = super(LayerContrastiveEncoder, self).get_config()
		config["input_width"] = self.param_input_width
		config["input_height"] = self.param_input_height
		config["input_channels"] = self.param_input_channels
		config["feature_dim"] = self.param_feature_dim
		return config
	
	def build(self, input_shape):
		# print("LAYER:build input_shape", input_shape)
		super().build(input_shape=input_shape[0])
		self.embedding.build(input_shape=tf.TensorShape([ *self.embedding_input_shape ]))
	
	def call(self, input_thing):
		result = self.encoder(input_thing)
		
		shape_ksize = result.shape[1]
		result = tf.nn.avg_pool(result, ksize=shape_ksize, strides=1, padding="VALID")
		
		target_shape = [ -1, result.shape[-1] ]
		result = self.embedding(tf.reshape(result, target_shape))
		return result