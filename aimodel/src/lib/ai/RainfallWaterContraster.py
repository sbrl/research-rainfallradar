import os
import json

from loguru import logger
import tensorflow as tf

from ..dataset.batched_iterator import batched_iterator

# from ..io.find_paramsjson import find_paramsjson
# from ..io.readfile import readfile
from ..io.writefile import writefile

from .model_rainfallwater_contrastive import model_rainfallwater_contrastive
from .helpers import make_callbacks
from .helpers import summarywriter
from .components.LayerContrastiveEncoder import LayerContrastiveEncoder
from .components.LayerConvNeXtGamma import LayerConvNeXtGamma
from .components.LayerCheeseMultipleOut import LayerCheeseMultipleOut


class RainfallWaterContraster(object):
	def __init__(self, dir_output=None, filepath_checkpoint=None, epochs=50, batch_size=64, **kwargs):
		super(RainfallWaterContraster, self).__init__()
		
		self.dir_output = dir_output
		self.epochs = epochs
		self.kwargs = kwargs
		self.batch_size = batch_size
		
		
		if filepath_checkpoint is None:
			if self.dir_output is None:
				raise Exception("Error: dir_output was not specified, and since no checkpoint was loaded training mode is activated.")
			if not os.path.exists(self.dir_output):
				os.mkdir(self.dir_output)
			
			self.filepath_summary = os.path.join(self.dir_output, "summary.txt")
			
			writefile(self.filepath_summary, "") # Empty the file ahead of time
			self.make_model()
			
			summarywriter(self.model, self.filepath_summary, append=True)
			writefile(os.path.join(self.dir_output, "params.json"), json.dumps(self.get_config()))
		else:	
			self.load_model(filepath_checkpoint)
	
	def get_config(self):
		return {
			"epochs": self.epochs,
			"batch_size": self.batch_size,
			**self.kwargs
		}
	
	@staticmethod
	def from_checkpoint(filepath_checkpoint, **hyperparams):
		logger.info(f"Loading from checkpoint: {filepath_checkpoint}")
		return RainfallWaterContraster(filepath_checkpoint=filepath_checkpoint, **hyperparams)
	
	
	def make_model(self):
		self.model, self.model_predict = model_rainfallwater_contrastive(
			batch_size=self.batch_size,
			summary_file=self.filepath_summary,
			**self.kwargs
		)
	
	
	def load_model(self, filepath_checkpoint):
		"""
		Loads a saved model from the given filename.
		filepath_checkpoint (string): The filepath to load the saved model from.
		"""
		
		self.model_predict = tf.keras.models.load_model(filepath_checkpoint, custom_objects={
			"LayerContrastiveEncoder": LayerContrastiveEncoder,
			"LayerConvNeXtGamma": LayerConvNeXtGamma,
			"LayerCheeseMultipleOut": LayerCheeseMultipleOut
		})
	
	
	
	def train(self, dataset_train, dataset_validate):
		return self.model.fit(
			dataset_train,
			validation_data=dataset_validate,
			epochs=self.epochs,
			callbacks=make_callbacks(self.dir_output, self.model_predict),
			# steps_per_epoch=10 # For testing
		)
	
	def embed(self, dataset):
		i_batch = -1
		for batch in batched_iterator(dataset, tensors_in_item=2, batch_size=self.batch_size):
			i_batch += 1
			rainfall = self.model_predict(batch[0], training=False) # ((rainfall, water), dummy_label)
			rainfall = tf.unstack(rainfall, axis=0)
			water = tf.unstack(batch[1], axis=0)
			for step_rainfall, step_water in zip(rainfall, water):
				yield step_rainfall, step_water
		
	
	# def embed_rainfall(self, dataset):
	# 	result = []
	# 	for batch in dataset:
	# 		result_batch = self.model_predict(batch)
	# 		result.extend(tf.unstack(result_batch, axis=0))
	# 	return result