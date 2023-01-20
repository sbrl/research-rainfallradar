#!/usr/bin/env python3
from loguru import logger

import os

import tensorflow as tf

from lib.dataset.dataset_encoderonly import dataset_encoderonly
from lib.ai.components.convnext import make_convnext
from lib.ai.helpers.summarywriter import summarywriter

# ███████ ███    ██ ██    ██
# ██      ████   ██ ██    ██
# █████   ██ ██  ██ ██    ██
# ██      ██  ██ ██  ██  ██
# ███████ ██   ████   ████

# TODO: env vars & settings here
DIRPATH_INPUT	= os.environ["DIRPATH_INPUT"]
DIRPATH_OUTPUT	= os.environ["DIRPATH_OUTPUT"]
PATH_HEIGHTMAP	= os.environ["PATH_HEIGHTMAP"]			if "PATH_HEIGHTMAP"		in os.environ else None
CHANNELS		= os.environ["CHANNELS"]				if "CHANNELS"			in os.environ else 8

BATCH_SIZE		= int(os.environ["BATCH_SIZE"])			if "BATCH_SIZE"			in os.environ else 64
WINDOW_SIZE		= int(os.environ["WINDOW_SIZE"])		if "WINDOW_SIZE"		in os.environ else 33
STEPS_PER_EPOCH = int(os.environ["STEPS_PER_EPOCH"])	if "STEPS_PER_EPOCH"	in os.environ else None

logger.info("Encoder-only rainfall radar TEST")
logger.info(f"> DIRPATH_INPUT {DIRPATH_INPUT}")
logger.info(f"> DIRPATH_OUTPUT {DIRPATH_OUTPUT}")
logger.info(f"> PATH_HEIGHTMAP {PATH_HEIGHTMAP}")
logger.info(f"> CHANNELS {CHANNELS}")
logger.info(f"> BATCH_SIZE {BATCH_SIZE}")
logger.info(f"> WINDOW_SIZE {WINDOW_SIZE}")
logger.info(f"> STEPS_PER_EPOCH {STEPS_PER_EPOCH}")


if not os.path.exists(DIRPATH_OUTPUT):
	os.makedirs(os.path.join(DIRPATH_OUTPUT, "checkpoints"))


# ██████   █████  ████████  █████  ███████ ███████ ████████
# ██   ██ ██   ██    ██    ██   ██ ██      ██         ██
# ██   ██ ███████    ██    ███████ ███████ █████      ██
# ██   ██ ██   ██    ██    ██   ██      ██ ██         ██
# ██████  ██   ██    ██    ██   ██ ███████ ███████    ██

dataset_train, dataset_validate = dataset_encoderonly(
	dirpath_input=DIRPATH_INPUT,
	filepath_heightmap=PATH_HEIGHTMAP,
	batch_size=BATCH_SIZE,
	windowsize=WINDOW_SIZE,
	rainfall_scale_up=2
)


# ███    ███  ██████  ██████  ███████ ██
# ████  ████ ██    ██ ██   ██ ██      ██
# ██ ████ ██ ██    ██ ██   ██ █████   ██
# ██  ██  ██ ██    ██ ██   ██ ██      ██
# ██      ██  ██████  ██████  ███████ ███████

def make_encoderonly(windowsize, channels, encoder="convnext", water_bins=2):
	if encoder == "convnext":
		model = make_convnext(input_shape=(windowsize, windowsize, channels), num_classes=water_bins, **kwargs)
	elif encoder == "resnet":
		layer_in = tf.keras.Input(shape=(windowsize, windowsize, channels))
		layer_next = tf.keras.applications.resnet50.ResNet50(
			weights=None,
			include_top=True,
			classes=water_bins,
			input_tensor=layer_in,
			pooling="max",
		)
		
		model = tf.keras.Model(
			inputs = layer_in,
			outputs = layer_next
		)
	else:
		raise Exception(f"Error: Unknown encoder '{encoder}' (known encoders: convnext, resnet).")
	
	model.compile(
		optimizer="Adam",
		loss = tf.keras.losses.SparseCategoricalCrossentropy(),
		metrics = [
			tf.keras.metrics.SparseCategoricalAccuracy()
		]
	)
	
	return model


model = make_encoderonly(
	windowsize=WINDOW_SIZE,
	channels=CHANNELS
)
summarywriter(model, os.path.join(DIRPATH_OUTPUT, "summary.txt"))

# ████████ ██████   █████  ██ ███    ██ ██ ███    ██  ██████  
#    ██    ██   ██ ██   ██ ██ ████   ██ ██ ████   ██ ██       
#    ██    ██████  ███████ ██ ██ ██  ██ ██ ██ ██  ██ ██   ███ 
#    ██    ██   ██ ██   ██ ██ ██  ██ ██ ██ ██  ██ ██ ██    ██ 
#    ██    ██   ██ ██   ██ ██ ██   ████ ██ ██   ████  ██████  

history = model.fit(dataset_train,
	validation_data=dataset_validate,
	epochs=25,

	callbacks=[
		tf.keras.callbacks.CSVLogger(
			filename=os.path.join(DIRPATH_OUTPUT, "metrics.tsv"),
			separator="\t"
		),
		CallbackCustomModelCheckpoint(
			model_to_checkpoint=model,
			filepath=os.path.join(
				DIRPATH_OUTPUT,
				"checkpoints"
				"checkpoint_e{epoch:d}_loss{loss:.3f}.hdf5"
			),
			monitor="loss"
		),
	],
	steps_per_epoch=STEPS_PER_EPOCH,
)
logger.info(">>> Training complete")

logger.info(">>> Plotting graphs")

plt.plot(history.history["loss"])
plt.title("Training Loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.savefig(os.path.join(DIRPATH_OUTPUT, "loss.png"))
plt.close()

plt.plot(history.history["accuracy"])
plt.title("Training Accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.savefig(os.path.join(DIRPATH_OUTPUT, "acc.png"))
plt.close()

plt.plot(history.history["val_loss"])
plt.title("Validation Loss")
plt.ylabel("val_loss")
plt.xlabel("epoch")
plt.savefig(os.path.join(DIRPATH_OUTPUT, "val_loss.png"))
plt.close()

plt.plot(history.history["val_accuracy"])
plt.title("Validation Accuracy")
plt.ylabel("val_accuracy")
plt.xlabel("epoch")
plt.savefig(os.path.join(DIRPATH_OUTPUT, "val_acc.png"))
plt.close()


# ██████  ██████  ███████ ██████  ██  ██████ ████████ ██  ██████  ███    ██
# ██   ██ ██   ██ ██      ██   ██ ██ ██         ██    ██ ██    ██ ████   ██
# ██████  ██████  █████   ██   ██ ██ ██         ██    ██ ██    ██ ██ ██  ██
# ██      ██   ██ ██      ██   ██ ██ ██         ██    ██ ██    ██ ██  ██ ██
# ██      ██   ██ ███████ ██████  ██  ██████    ██    ██  ██████  ██   ████


logger.info("Predictions coming soon.")
# TODO FILL THIS IN


# █████ █████ █████ █████ █████ █████ █████ █████ █████
