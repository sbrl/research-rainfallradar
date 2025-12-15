import tensorflow as tf

from lib.io.handle_open import handle_open

class CallbackAdvancedCsv(tf.keras.callbacks.Callback):
	def __init__(self, filepath:str, dirpath_weighted:str|None=None, n_epochs:int=1, separator:str="\t", **kwargs) -> None:
		super(CallbackAdvancedCsv, self).__init__(**kwargs)
		
		self.n_epochs = n_epochs
		self.separator = separator
		
		self.handle = handle_open(filepath, "w")
		
		
		self.epochs_seen = 0
		self.keys = None
	
	def write_header(self, logs): # logs = metrics
		self.keys = logs.keys()
		self.keys = sorted(self.keys)
		self.handle.write("\t".join(self.keys)+"\n")
			
	def on_epoch_end(self, epoch, logs=None): # logs = metrics
		if self.epochs_seen == 0:
			self.write_header(logs)
		
		if self.epochs_seen % self.n_epochs == 0:
			
			self.handle.write(self.separator.join([str(logs[key]) for key in self.keys]) + "\n")
		
		self.epochs_seen += 1
