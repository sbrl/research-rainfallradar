import tensorflow as tf

def get_from_batched_dataset(dataset, count):
    result = []
    for batched in dataset:
        items_input = tf.unstack(batched[0], axis=0)
        items_label = tf.unstack(batched[1], axis=0)
        for item in zip(items_input, items_label):
            result.append(item)
            if len(result) >= count:
                return result
