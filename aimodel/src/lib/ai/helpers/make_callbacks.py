import os

import tensorflow as tf

from ..components.CallbackCustomModelCheckpoint import CallbackCustomModelCheckpoint

def make_callbacks(dirpath, model_predict):
	dirpath_checkpoints = os.path.join(dirpath, "checkpoints")
	filepath_metrics = os.path.join(dirpath, "metrics.tsv")
	
	if not os.path.exists(dirpath_checkpoints):
		os.mkdir(dirpath_checkpoints)
	
	return [
		CallbackCustomModelCheckpoint(
			model_to_checkpoint=model_predict,
			filepath=os.path.join(
				dirpath_checkpoints,
				"checkpoint_weights_e{epoch:d}_loss{loss:.3f}.hdf5"
			),
			monitor="loss"
		),
		tf.keras.callbacks.CSVLogger(
			filename=filepath_metrics,
			separator="\t"
		),
		tf.keras.callbacks.ProgbarLogger()
	]