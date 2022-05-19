"use strict";

// import tf from '@tensorflow/tfjs-node';
import tf from '@tensorflow/tfjs-node-gpu';

export default async function array2d_pool(channels, operator="max") {
	if(operator !== "max")
		throw new Error(`Error: Unknown operator '${operator}. At present only the 'max' operator is supported.`);
	
	// This is rather a hack to save time. Tensorflow.js is not needed here, but may result in increased speed. It may be worth rolling this out to the rest of the codebase, thinking about it. While Tensorflow.js hasmany bugs, this only extends to the machine learning / loss functions / models etc and not the 
	const result_tensor = tf.tidy(() => {
		const tensor = tf.tensor(channels);
		return tf.max(tensor, 0, false);
	});
	
	const result_array = await result_tensor.array();
	result_tensor.dispose();
	
	return result_array;
}
