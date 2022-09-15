import tensorflow as tf


def batched_iterator(dataset, tensors_in_item=1, batch_size=64):
	acc = [ [] for _ in range(tensors_in_item) ]
	i_item = 0
	for item in dataset:
		i_item += 1
		
		if tensors_in_item == 1:
			item = [ item ]
			
		
		for i_tensor, tensor in enumerate(item):
			acc[i_tensor].append(tensor)
		
		if i_item >= batch_size:
			yield [ tf.stack(tensors) for tensors in acc ]
			for arr in acc:
				arr.clear()
