import os
import io
import re
import sys
import json

import tensorflow as tf

from ..io.find_paramsjson import find_paramsjson
from ..io.readfile import readfile
from ..io.writefile import writefile

from .model_rainfallwater_contrastive import model_rainfallwater_contrastive
from .helpers import make_callbacks
from .helpers import summarywriter
from .components.LayerContrastiveEncoder import LayerContrastiveEncoder
from .components.LayerCheeseMultipleOut import LayerCheeseMultipleOut
from .helpers.summarywriter import summarywriter

class RainfallWaterContraster(object):
	def __init__(self, dir_output=None, filepath_checkpoint=None, epochs=50, batch_size=64, **kwargs):
		super(RainfallWaterContraster, self).__init__()
		
		self.dir_output = dir_output
		self.epochs = epochs
		self.kwargs = kwargs
		self.batch_size = batch_size
		
		
		if filepath_checkpoint == None:
			if self.dir_output == None:
				raise Exception("Error: dir_output was not specified, and since no checkpoint was loaded training mode is activated.")
			if not os.path.exists(self.dir_output):
				os.mkdir(self.dir_output)
			
			self.filepath_summary = os.path.join(self.dir_output, "summary.txt")
			
			writefile(self.filepath_summary, "") # Empty the file ahead of time
			self.model = self.make_model()
			
			summarywriter(self.model, self.filepath_summary, append=True)
			writefile(os.path.join(self.dir_output, "params.json"), json.dumps(self.get_config()))
		else:	
			self.model = self.load_model(filepath_checkpoint)
	
	def get_config(self):
		return {
			"epochs": self.epochs,
			"batch_size": self.batch_size,
			**self.kwargs
		}
	
	@staticmethod
	def from_checkpoint(filepath_checkpoint, filepath_hyperparams=None):
		if filepath_checkpoint is None:
			filepath_checkpoint = find_paramsjson(filepath_checkpoint)
		hyperparams = json.loads(readfile(filepath_hyperparams))
		return RainfallWaterContraster(filepath_checkpoint=filepath_checkpoint, **hyperparams)
	
	
	def make_model(self):
		return model_rainfallwater_contrastive(
			batch_size=self.batch_size,
			summary_file=self.filepath_summary,
			**self.kwargs
		)
	
	
	def load_model(self, filepath_checkpoint):
		"""
		Loads a saved model from the given filename.
		filepath_checkpoint (string): The filepath to load the saved model from.
		"""
		
		return tf.keras.models.load_model(filepath_checkpoint, custom_objects={
			"LayerContrastiveEncoder": LayerContrastiveEncoder,
			"LayerCheeseMultipleOut": LayerCheeseMultipleOut
		})
	
	
	
	def train(self, dataset_train, dataset_validate):
		return self.model.fit(
			dataset_train,
			validation_data=dataset_validate,
			epochs=self.epochs,
			callbacks=make_callbacks(self.dir_output)
		)
	
	def embed(self, dataset):
		result = []
		i_batch = -1
		for batch in dataset:
			i_batch += 1
			result_batch = self.model(batch[0])
			rainfall, water = tf.unstack(result_batch, axis=-2)
			
			rainfall = tf.unstack(rainfall, axis=0)
			water = tf.unstack(water, axis=0)
			
			result.extend(zip(rainfall, water))
		
		return result