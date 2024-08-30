import tensorflow as tf
from loguru import logger


class CallbackExtraValidation(tf.keras.callbacks.Callback):
	"""
	A custom (keras) callback that to evaluate metrics on additional datasets during training.
	
	These are passed back to Tensorflow/Keras by ~~abusing~~ updating the logs dictionary that's passed to us. If you update it with more metrics, then they get fed into the regular Tensorflow logging system :D
	
	IMPORTANT: This MUST be the FIRST callback in the list! Otherwise it won't be executed before e.g. `tf.kkeras.callbacks.CSVLogger`.
	
	TODO note to self blog about this because this was not as easy to figure out as it appears.
	
	Ref kudos to <https://stackoverflow.com/a/47738812/1460422>, but you don't need to go to all that trouble :P
	
	Args:
		datasets (dict): A dictionary mapping dataset names to TensorFlow Dataset
			objects.
		verbose (str, optional): The verbosity level for the dataset evaluations. Basically the same as `verbose=VALUE` on `tf.keras.Model.fit()`. Default: `"auto"`.
	"""
	
	def __init__(self, datasets, verbose="auto"):
		super(CallbackExtraValidation, self).__init__()
		# self.model = model # apparently this exists by default??
		self.datasets = datasets
		self.verbose = verbose

	def on_epoch_end(self, epoch, logs=None):
		if logs == None:
			logger.warning(
				"[CallbackExtraValidation] logs is None! Can't do anything here.")
			return False

		for name, dataset in self.datasets.items():
			if dataset is None:
				logger.info(f"Skipping extra dataset {name} because it's None")
				continue

			metrics = self.model.evaluate(
				dataset, verbose=self.verbose, return_dict=True)

			for metric_name, metric_value in metrics.items():
				logs[f"{name}_{metric_name}"] = metric_value

			print(metrics)
