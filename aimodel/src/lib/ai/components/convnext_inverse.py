
import tensorflow as tf

from .convnext import add_convnext_block

depths_dims = dict(
	# architectures from: https://github.com/facebookresearch/ConvNeXt
	# A ConvNet for the 2020s: https://arxiv.org/abs/2201.03545
	convnext_i_xtiny  = (dict(depths=[3, 6, 3, 3],    dims=[528, 264, 132, 66])),
	convnext_i_tiny   = (dict(depths=[3, 9, 3, 3],    dims=[768, 384, 192, 96])),
	convnext_i_small  = (dict(depths=[3, 27, 3, 3],   dims=[768, 384, 192, 96])),
	convnext_i_base   = (dict(depths=[3, 27, 3, 3],   dims=[1024, 512, 256, 128])),  
	convnext_i_large  = (dict(depths=[3, 27, 3, 3],   dims=[1536, 768, 384, 192])),
	convnext_i_xlarge = (dict(depths=[3, 27, 3, 3],   dims=[2048, 1024, 512, 256])),
)


def do_convnext_inverse(layer_in, arch_name="convnext_tiny"):
	return convnext_inverse(layer_in,
		depths=depths_dims[arch_name]["depths"],
		dims=depths_dims[arch_name]["dims"]
	)


def convnext_inverse(layer_in, depths, dims):
	layer_next = layer_in
	
	i = 0
	for depth, dim in zip(depths, dims):
		layer_next = block_upscale(layer_next, i, depth=depth, dim=dim)
		i += 1
	
	return layer_next


def block_upscale(layer_in, block_number, depth, dim):
	layer_next = layer_in
	
	layer_next = tf.keras.layers.Conv2DTranspose(
		name=f"cns.stage{block_number}.end.convtp",
		filters=dim,
		kernel_size=4,
		stride=2
	)(layer_next)
	layer_next = tf.keras.layers.LayerNormalization(name=f"cns.stage{block_number}.end.norm", epsilon=1e-6)(layer_next)
	
	for i in range(depth):
		layer_next = add_convnext_block(layer_next, dim=dim, prefix=f"cns.stage{block_number}.block.{i}")
	
	return layer_next
