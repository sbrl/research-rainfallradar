import json

import tensorflow as tf
from ..io.readfile import readfile

def parse_heightmap(filepath_heightmap):
	obj = json.loads(readfile(filepath_heightmap))
	
	result = tf.constant(obj["data"])
	result = tf.transpose(result, [1,0]) # [ height, width ] → [ width, height ]
	
	return result