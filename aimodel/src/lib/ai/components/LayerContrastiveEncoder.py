import tensorflow as tf

# from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from ..helpers.summarywriter import summarylogger
from .convnext import make_convnext

class LayerContrastiveEncoder(tf.keras.layers.Layer):
	
	def __init__(self, input_width, input_height, channels, feature_dim=2048, **kwargs):
		"""Creates a new contrastive learning encoder layer.
		Note that the input format MUST be channels_last. This is because Tensorflow/Keras' Dense layer does NOT support specifying an axis. Go complain to them, not me.
		While this is intended for contrastive learning, this can (in theory) be used anywhere as it's just a generic wrapper layer.
		The key feature here is that it does not care about the input size or the number of channels.
		Currently it uses a ConvNeXt internally, but an upgrade to Tensorflow's internal ConvNeXt implementation is planned once it comes out of nightly and into stable.
		
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
		
		"""The main ConvNeXt model that forms the encoder.
		"""
		self.encoder = make_convnext(
			input_shape				= (self.param_input_width, self.param_input_height, self.param_channels),
			classifier_activation	= tf.nn.relu, # this is not actually a classifier, but rather a feature encoder
			num_classes				= self.param_feature_dim # size of the feature dimension, see the line above this one
		)
		# """Small sequential stack of layers that control the size of the outputted feature dimension.
		# """
		# self.embedding = tf.keras.layers.Dense(self.param_feature_dim)
		
		summarylogger(self.encoder)
	
	def get_config(self):
		config = super(LayerContrastiveEncoder, self).get_config()
		config["input_width"] = self.param_input_width
		config["input_height"] = self.param_input_height
		config["input_channels"] = self.param_input_channels
		config["feature_dim"] = self.param_feature_dim
		return config
	
	# def build(self, input_shape):
	# 	# print("LAYER:build input_shape", input_shape)
	# 	super().build(input_shape=input_shape[0])
	# 	self.embedding.build(input_shape=tf.TensorShape([ *self.embedding_input_shape ]))
	
	def call(self, input_thing):
		result = self.encoder(input_thing)
		
		# The encoder is handled by the ConvNeXt model \o/
		# shape_ksize = result.shape[1]
		# result = tf.nn.avg_pool(result, ksize=shape_ksize, strides=1, padding="VALID")
		
		# target_shape = [ -1, result.shape[-1] ]
		# result = self.embedding(tf.reshape(result, target_shape))
		return result