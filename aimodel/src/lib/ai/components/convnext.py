from unicodedata import name
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


# Code from https://github.com/leanderme/ConvNeXt-Tensorflow/blob/main/ConvNeXt.ipynb
from .LayerConvNeXtGamma import LayerConvNeXtGamma

kernel_initial = tf.keras.initializers.TruncatedNormal(stddev=0.2)
bias_initial   = tf.keras.initializers.Constant(value=0)

depths_dims = dict(
	convnext_xtiny  = (dict(depths=[3, 3, 6, 3],    dims=[66, 132, 264, 528])),
	# architectures from: https://github.com/facebookresearch/ConvNeXt
	# A ConvNet for the 2020s: https://arxiv.org/abs/2201.03545
	convnext_tiny   = (dict(depths=[3, 3, 9, 3],    dims=[96, 192, 384, 768])),
	convnext_small  = (dict(depths=[3, 3, 27, 3],   dims=[96, 192, 384, 768])),
	convnext_base   = (dict(depths=[3, 3, 27, 3],   dims=[128, 256, 512, 1024])),  
	convnext_large  = (dict(depths=[3, 3, 27, 3],   dims=[192, 384, 768, 1536])),
	convnext_xlarge = (dict(depths=[3, 3, 27, 3],   dims=[256, 512, 1024, 2048])),
)

__convnext_next_model_number = 0

def make_convnext(input_shape, arch_name="convnext_tiny", **kwargs):
	"""Makes a ConvNeXt model.
	Returns a tf.keras.Model.
	Args:
		input_shape (int[]): The input shape of the tensor that will be fed to the ConvNeXt model. This is necessary as we make the model using the functional API and thus we need to make an Input layer.
		arch_name (str, optional): The name of the preset ConvNeXt model architecture to use. Defaults to "convnext_tiny".
	"""
	global __convnext_next_model_number
	
	layer_in = tf.keras.layers.Input(
		shape = input_shape
	)
	layer_out = convnext(layer_in, **depths_dims[arch_name], **kwargs)
	result = tf.keras.Model(
		name=f"convnext{__convnext_next_model_number}",
		inputs	= layer_in,
		outputs	= layer_out
	)
	__convnext_next_model_number += 1
	return result


def convnext(
	x,
	include_top				= True,
	num_classes				= 1000,
	depths					= [3, 3, 9, 3],
	dims					= [96, 192, 384, 768],
	drop_path_rate			= 0.,
	classifier_activation	= 'softmax'
	# Note that we CAN'T add data_format here, 'cause Dense doesn't support specifying the axis
):
	
	assert len(depths) == len(dims)
	
	def forward_features(x):
		i = 0
		for depth, dim in zip(depths, dims):
			
			if i == 0:
				x = tf.keras.layers.Conv2D(
					dim,
					kernel_size	= 4,
					strides		= 4,
					padding		= "valid",
					name		= "downsample_layers.0.0_conv"
				)(x)
				x = tf.keras.layers.LayerNormalization(
					epsilon	= 1e-6,
					name	= "downsample_layers.0.0_norm"
				)(x)
			else:
				x = tf.keras.layers.LayerNormalization(
					epsilon = 1e-6,
					name    = "stages." + str(i) + "." + str(k) + ".downsample_norm"
				)(x)
				x = tf.keras.layers.Conv2D(
					dim,
					kernel_size			= 2,
					strides				= 2,
					padding				='same',
					kernel_initializer	= kernel_initial,
					bias_initializer	= bias_initial,
					name				= "stages." + str(i) + "." + str(k) + ".downsample_conv"
				)(x)
			
			
			for k in range(depth):
				x = add_convnext_block(
					x,
					dim,
					drop_path_rate,
					prefix		= "stages." + str(i) + "." + str(k),
				)        
			i = i +1

		return x
	
	x = forward_features(x)
	
	if include_top:
		x = tf.keras.layers.GlobalAveragePooling2D(
			name		= 'avg'
		)(x)
		x = tf.keras.layers.LayerNormalization(
			epsilon		= 1e-6,
			name		= "norm",
		)(x)
		
		
		x = tf.keras.layers.Dense(
			num_classes,
			activation         = classifier_activation,
			kernel_initializer = kernel_initial,
			bias_initializer   = bias_initial,
			name               = "head"
		)(x)
	else:
		x = tf.keras.layers.GlobalAveragePooling2D(name='avg')(x)
		x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="norm")(x)
	
	return x


def add_convnext_block(y, dim, drop_prob=0, prefix=""):
	skip = y
		
	y = tf.keras.layers.DepthwiseConv2D(
		kernel_size=7,
		padding='same',
		name = f'{prefix}.dwconv'
	)(y)
	
	y = tf.keras.layers.LayerNormalization(
		epsilon=1e-6,
		name=f'{prefix}.norm'
	)(y)
	
	y = tf.keras.layers.Dense(
		4 * dim,
		name=f'{prefix}.pwconv1'
	)(y)
	
   
	y = tf.keras.layers.Activation(
		'gelu',
		name=f'{prefix}.act'
	)(y)
	
	y = tf.keras.layers.Dense(
		dim,
		name=f'{prefix}.pwconv2'
	)(y)
	
	y = LayerConvNeXtGamma(
		const_val = 1e-6,
		dim       = dim,
		name      = f'{prefix}.gamma'
	)(y)
	
	y = tfa.layers.StochasticDepth(
		drop_prob,
		name = f'{prefix}.drop_path'
	)([skip, y])
	
	return y
