"use strict";

// import tf from '@tensorflow/tfjs-node';
import tf from '@tensorflow/tfjs-node-gpu';

export default async function array2d_pool(channels, operator) {
	// This is rather a hack to save time. Tensorflow.js is not needed here, but may result in increased speed. It may be worth rolling this out to the rest of the codebase, thinking about it. While Tensorflow.js hasmany bugs, this only extends to the machine learning / loss functions / models etc and not the 
	const result_tensor = tf.tidy(() => {
		const tensor = tf.tensor(channels);
		console.log(`DEFAULT array2d_pool tensor shape:`, tensor);
		return tf.max(tensor, 0, false);
	});
	
	const result_array = await result.array();
	result_tensor.dispose();
	
	return result_array;
}
