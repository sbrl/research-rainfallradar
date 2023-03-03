#!/usr/bin/env python3
# @source https://keras.io/examples/vision/deeplabv3_plus/
# Required dataset: https://drive.google.com/uc?id=1B9A9UCJYMwTL4oBEo4RZfbMZMaZhKJaz [instance-level-human-parsing.zip]

from datetime import datetime
from loguru import logger
from lib.ai.helpers.summarywriter import summarywriter
from lib.ai.components.CallbackCustomModelCheckpoint import CallbackCustomModelCheckpoint

import os
import cv2
import numpy as np
from glob import glob
from scipy.io import loadmat
import matplotlib.pyplot as plt

import tensorflow as tf

from lib.dataset.dataset_mono import dataset_mono
from lib.ai.components.LossCrossEntropyDice import LossCrossEntropyDice
from lib.ai.components.MetricDice import MetricDice
from lia.ai.components.MetricSensitivity import MetricSensitivity
from lib.ai.components.MetricSpecificity import MetricSpecificity

time_start = datetime.now()
logger.info(f"Starting at {str(datetime.now().isoformat())}")


# ███████ ███    ██ ██    ██ ██ ██████   ██████  ███    ██ ███    ███ ███████ ███    ██ ████████
# ██      ████   ██ ██    ██ ██ ██   ██ ██    ██ ████   ██ ████  ████ ██      ████   ██    ██
# █████   ██ ██  ██ ██    ██ ██ ██████  ██    ██ ██ ██  ██ ██ ████ ██ █████   ██ ██  ██    ██
# ██      ██  ██ ██  ██  ██  ██ ██   ██ ██    ██ ██  ██ ██ ██  ██  ██ ██      ██  ██ ██    ██
# ███████ ██   ████   ████   ██ ██   ██  ██████  ██   ████ ██      ██ ███████ ██   ████    ██

IMAGE_SIZE = int(os.environ["IMAGE_SIZE"]) if "IMAGE_SIZE" in os.environ else 128 # was 512; 128 is the highest power of 2 that fits the data
BATCH_SIZE = int(os.environ["BATCH_SIZE"]) if "BATCH_SIZE" in os.environ else 64
NUM_CLASSES = 2
DIR_RAINFALLWATER = os.environ["DIR_RAINFALLWATER"]
PATH_HEIGHTMAP = os.environ["PATH_HEIGHTMAP"]
PATH_COLOURMAP = os.environ["PATH_COLOURMAP"]
STEPS_PER_EPOCH = int(os.environ["STEPS_PER_EPOCH"]) if "STEPS_PER_EPOCH" in os.environ else None
REMOVE_ISOLATED_PIXELS = False if "NO_REMOVE_ISOLATED_PIXELS" in os.environ else True
EPOCHS = int(os.environ["EPOCHS"]) if "EPOCHS" in os.environ else 25
LOSS = os.environ["LOSS"] if "LOSS" in os.environ else "cross-entropy"
LEARNING_RATE = float(os.environ["LEARNING_RATE"]) if "LEARNING_RATE" in os.environ else 0.001

DIR_OUTPUT=os.environ["DIR_OUTPUT"] if "DIR_OUTPUT" in os.environ else f"output/{datetime.utcnow().date().isoformat()}_deeplabv3plus_rainfall_TEST"

PATH_CHECKPOINT = os.environ["PATH_CHECKPOINT"] if "PATH_CHECKPOINT" in os.environ else None
PREDICT_COUNT = int(os.environ["PREDICT_COUNT"]) if "PREDICT_COUNT" in os.environ else 4

# ~~~

if not os.path.exists(DIR_OUTPUT):
	os.makedirs(os.path.join(DIR_OUTPUT, "checkpoints"))

# ~~~

logger.info("DeepLabV3+ rainfall radar TEST")
for env_name in [ "BATCH_SIZE","NUM_CLASSES", "DIR_RAINFALLWATER", "PATH_HEIGHTMAP", "PATH_COLOURMAP", "STEPS_PER_EPOCH", "REMOVE_ISOLATED_PIXELS", "EPOCHS", "LOSS", "LEARNING_RATE", "DIR_OUTPUT", "PATH_CHECKPOINT", "PREDICT_COUNT" ]:
	logger.info(f"> {env_name} {str(globals()[env_name])}")


# ██████   █████  ████████  █████  ███████ ███████ ████████ 
# ██   ██ ██   ██    ██    ██   ██ ██      ██         ██    
# ██   ██ ███████    ██    ███████ ███████ █████      ██    
# ██   ██ ██   ██    ██    ██   ██      ██ ██         ██    
# ██████  ██   ██    ██    ██   ██ ███████ ███████    ██    

dataset_train, dataset_validate = dataset_mono(
	dirpath_input=DIR_RAINFALLWATER,
	batch_size=BATCH_SIZE,
	water_threshold=0.1,
	rainfall_scale_up=2, # done BEFORE cropping to the below size
	output_size=IMAGE_SIZE,
	input_size="same",
	filepath_heightmap=PATH_HEIGHTMAP,
	do_remove_isolated_pixels=REMOVE_ISOLATED_PIXELS
)

logger.info("Train Dataset:", dataset_train)
logger.info("Validation Dataset:", dataset_validate)


# ███    ███  ██████  ██████  ███████ ██     
# ████  ████ ██    ██ ██   ██ ██      ██     
# ██ ████ ██ ██    ██ ██   ██ █████   ██     
# ██  ██  ██ ██    ██ ██   ██ ██      ██     
# ██      ██  ██████  ██████  ███████ ███████


if PATH_CHECKPOINT is None:
	def convolution_block(
		block_input,
		num_filters=256,
		kernel_size=3,
		dilation_rate=1,
		padding="same",
		use_bias=False,
	):
		x = tf.keras.layers.Conv2D(
			num_filters,
			kernel_size=kernel_size,
			dilation_rate=dilation_rate,
			padding="same",
			use_bias=use_bias,
			kernel_initializer=tf.keras.initializers.HeNormal(),
		)(block_input)
		x = tf.keras.layers.BatchNormalization()(x)
		return tf.nn.relu(x)


	def DilatedSpatialPyramidPooling(dspp_input):
		dims = dspp_input.shape
		x = tf.keras.layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
		x = convolution_block(x, kernel_size=1, use_bias=True)
		out_pool = tf.keras.layers.UpSampling2D(
			size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
		)(x)
		
		out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
		out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
		out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
		out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)
		
		x = tf.keras.layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
		output = convolution_block(x, kernel_size=1)
		return output
	
	
	def DeeplabV3Plus(image_size, num_classes, num_channels=3):
		model_input = tf.keras.Input(shape=(image_size, image_size, num_channels))
		x = tf.keras.layers.UpSampling2D(size=2)(model_input)
		resnet50 = tf.keras.applications.ResNet50(
			weights="imagenet" if num_channels == 3 else None,
			include_top=False, input_tensor=x
		)
		x = resnet50.get_layer("conv4_block6_2_relu").output
		x = DilatedSpatialPyramidPooling(x)

		input_a = tf.keras.layers.UpSampling2D(
			size=(image_size // 4 // x.shape[1] * 2, image_size // 4 // x.shape[2] * 2), # <--- UPSAMPLE after pyramid
			interpolation="bilinear",
		)(x)
		input_b = resnet50.get_layer("conv2_block3_2_relu").output
		input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

		x = tf.keras.layers.Concatenate(axis=-1)([input_a, input_b])
		x = convolution_block(x)
		x = convolution_block(x)
		x = tf.keras.layers.UpSampling2D(
			size=(image_size // x.shape[1], image_size // x.shape[2]), # <--- UPSAMPLE at end
			interpolation="bilinear",
		)(x)
		model_output = tf.keras.layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
		return tf.keras.Model(inputs=model_input, outputs=model_output)


	model = DeeplabV3Plus(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES, num_channels=8)
	summarywriter(model, os.path.join(DIR_OUTPUT, "summary.txt"))
else:
	model = tf.keras.models.load_model(PATH_CHECKPOINT, custom_objects={
		# Tell Tensorflow about our custom layers so that it can deserialise models that use them
		"LossCrossEntropyDice": LossCrossEntropyDice,
		"MetricDice": MetricDice,
		"MetricSensitivity": MetricSensitivity,
		"MetricSpecificity": MetricSpecificity
	})



# ████████ ██████   █████  ██ ███    ██ ██ ███    ██  ██████  
#    ██    ██   ██ ██   ██ ██ ████   ██ ██ ████   ██ ██       
#    ██    ██████  ███████ ██ ██ ██  ██ ██ ██ ██  ██ ██   ███ 
#    ██    ██   ██ ██   ██ ██ ██  ██ ██ ██ ██  ██ ██ ██    ██ 
#    ██    ██   ██ ██   ██ ██ ██   ████ ██ ██   ████  ██████  

if PATH_CHECKPOINT is None:
	loss_fn = None
	if LOSS == "cross-entropy-dice":
		loss_fn = LossCrossEntropyDice()
	elif LOSS == "cross-entropy":
		loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
	else:
		raise Exception(f"Error: Unknown loss function '{LOSS}' (possible values: cross-entropy, cross-entropy-dice).")
	
	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
		loss=loss_fn,
		metrics=[
			"accuracy",
			MetricDice(),
			tf.keras.metrics.MeanIoU(num_classes=2),
			MetricSensitivity(), # How many true positives were accurately predicted
			MetricSpecificity() # How many true negatives were accurately predicted?
			# TODO: Add IoU, F1, Precision, Recall,  here. 
		],
	)
	logger.info(">>> Beginning training")
	history = model.fit(dataset_train,
		validation_data=dataset_validate,
		epochs=EPOCHS,
		callbacks=[
			tf.keras.callbacks.CSVLogger(
				filename=os.path.join(DIR_OUTPUT, "metrics.tsv"),
				separator="\t"
			),
			CallbackCustomModelCheckpoint(
				model_to_checkpoint=model,
				filepath=os.path.join(
					DIR_OUTPUT,
					"checkpoints",
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
	plt.savefig(os.path.join(DIR_OUTPUT, "loss.png"))
	plt.close()

	plt.plot(history.history["accuracy"])
	plt.title("Training Accuracy")
	plt.ylabel("accuracy")
	plt.xlabel("epoch")
	plt.savefig(os.path.join(DIR_OUTPUT, "acc.png"))
	plt.close()

	plt.plot(history.history["val_loss"])
	plt.title("Validation Loss")
	plt.ylabel("val_loss")
	plt.xlabel("epoch")
	plt.savefig(os.path.join(DIR_OUTPUT, "val_loss.png"))
	plt.close()

	plt.plot(history.history["val_accuracy"])
	plt.title("Validation Accuracy")
	plt.ylabel("val_accuracy")
	plt.xlabel("epoch")
	plt.savefig(os.path.join(DIR_OUTPUT, "val_acc.png"))
	plt.close()



# ██ ███    ██ ███████ ███████ ██████  ███████ ███    ██  ██████ ███████ 
# ██ ████   ██ ██      ██      ██   ██ ██      ████   ██ ██      ██      
# ██ ██ ██  ██ █████   █████   ██████  █████   ██ ██  ██ ██      █████   
# ██ ██  ██ ██ ██      ██      ██   ██ ██      ██  ██ ██ ██      ██      
# ██ ██   ████ ██      ███████ ██   ██ ███████ ██   ████  ██████ ███████ 

# Loading the Colormap
colormap = loadmat(
	PATH_COLOURMAP
)["colormap"]
colormap = colormap * 100
colormap = colormap.astype(np.uint8)


def infer(model, image_tensor):
	predictions = model.predict(tf.expand_dims((image_tensor), axis=0))
	predictions = tf.squeeze(predictions)
	predictions = tf.argmax(predictions, axis=2)
	return predictions


def decode_segmentation_masks(mask, colormap, n_classes):
	r = np.zeros_like(mask).astype(np.uint8)
	g = np.zeros_like(mask).astype(np.uint8)
	b = np.zeros_like(mask).astype(np.uint8)
	for l in range(0, n_classes):
		idx = mask == l
		r[idx] = colormap[l, 0]
		g[idx] = colormap[l, 1]
		b[idx] = colormap[l, 2]
	rgb = np.stack([r, g, b], axis=2)
	return rgb


def get_overlay(image, coloured_mask):
	image = tf.keras.preprocessing.image.array_to_img(image)
	image = np.array(image).astype(np.uint8)
	overlay = cv2.addWeighted(image, 0.35, coloured_mask, 0.65, 0)
	return overlay


def plot_samples_matplotlib(filepath, display_list):
	for i in range(len(display_list)):
		plt.subplot(1, len(display_list), i+1)
		if display_list[i].shape[-1] == 3:
			plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
		else:
			plt.imshow(display_list[i])
	plt.savefig(filepath, dpi=200)


def plot_predictions(filepath, input_items, colormap, model):
	i = 0
	for input_pair in input_items:
		prediction_mask = infer(image_tensor=input_pair[0], model=model)
		# label_colourmap = decode_segmentation_masks(input_pair[1], colormap, 2)
		prediction_colormap = decode_segmentation_masks(prediction_mask, colormap, 2)
		
		# print("DEBUG:plot_predictions INFER", str(prediction_mask.numpy().tolist()).replace("], [", "],\n["))
		
		plot_samples_matplotlib(
			filepath.replace("$$", str(i)),
			[
				# input_tensor,
				input_pair[1], #label_colourmap
				prediction_colormap
			]
		)
		i += 1

def get_from_batched(dataset, count):
	result = []
	for batched in dataset:
		items_input = tf.unstack(batched[0], axis=0)
		items_label = tf.unstack(batched[1], axis=0)
		for item in zip(items_input, items_label):
			result.append(item)
			if len(result) >= count:
				return result


plot_predictions(
	os.path.join(DIR_OUTPUT, "predict_train_$$.png"),
	get_from_batched(dataset_train, PREDICT_COUNT),
	colormap,
	model=model
)
plot_predictions(
	os.path.join(DIR_OUTPUT, "predict_validate_$$.png"),
	get_from_batched(dataset_validate, PREDICT_COUNT),
	colormap,
	model=model
)

logger.info(f"Complete at {str(datetime.now().isoformat())}, elapsed {str((datetime.now() - time_start).total_seconds())} seconds")
