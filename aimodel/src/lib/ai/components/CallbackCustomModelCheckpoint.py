import sys

from loguru import logger
import tensorflow as tf

class CallbackCustomModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, model_to_checkpoint, **kwargs) -> None:
        super().__init__()

        self.model_to_checkpoint = model_to_checkpoint
        self.checkpointer = tf.keras.callbacks.ModelCheckpoint(**kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logger.info("Saving checkpoint")
        print("DEBUG:CallbackCustomModelCheckpoint/on_epoch_end epoch", epoch, "logs", logs, file=sys.stderr)
        self.checkpointer.set_model(self.model_to_checkpoint)
        self.checkpointer.on_epoch_end(epoch=epoch, logs=logs)
        logger.info("Checkpoint saved successfully")