import tensorflow as tf

from lib.io.handle_open import handle_open

class CallbackNBatchCsv(tf.keras.callbacks.Callback):
	def __init__(self, filepath, n_batches=1, separator="\t", **kwargs) -> None:
		super().__init__(**kwargs)
		
		self.n_batches = n_batches
		self.separator = separator
		
		self.handle = handle_open(filepath)
		
		
		self.batches_seen = 0
		self.keys = None
	
	def write_header(self, logs): # logs = metrics
		self.keys = logs.keys()
		self.keys.sort()
		self.handle.write("\t".join(self.keys)+"\n")
			
	def on_batch_end(self, batch, logs=None): # logs = metrics
		if self.batches_seen == 0:
			self.write_header(logs)
		
		if self.batches_seen % self.n_batches == 0:
			self.handle.write(self.separator.join([str(logs[key]) for key in self.keys]) + "\n")
		
		self.batches_seen += 1
