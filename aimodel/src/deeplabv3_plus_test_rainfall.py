#!/usr/bin/env python3
# @source https://keras.io/examples/vision/deeplabv3_plus/
# Required dataset: https://drive.google.com/uc?id=1B9A9UCJYMwTL4oBEo4RZfbMZMaZhKJaz [instance-level-human-parsing.zip]

import io
import json
import math
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from loguru import logger
from scipy.io import loadmat

# import cv2 # optional import below in get_overlay

import lib.primitives.env as env
from lib.ai.components.CallbackCustomModelCheckpoint import CallbackCustomModelCheckpoint
from lib.ai.components.CallbackExtraValidation import CallbackExtraValidation
from lib.ai.components.LossCrossEntropyDice import LossCrossEntropyDice
from lib.ai.components.MetricDice import metric_dice_coefficient as dice_coefficient
from lib.ai.components.MetricMeanIoU import make_one_hot_mean_iou as mean_iou
from lib.ai.components.MetricSensitivity import make_sensitivity as sensitivity
from lib.ai.components.MetricSpecificity import specificity
from lib.ai.helpers.summarywriter import summarywriter
from lib.dataset.dataset_mono import dataset_mono, dataset_mono_predict

# from glob import glob

time_start = datetime.now()
logger.info(f"Starting at {str(datetime.now().isoformat())}")


# ███████ ███    ██ ██    ██ ██ ██████   ██████  ███    ██ ███    ███ ███████ ███    ██ ████████
# ██      ████   ██ ██    ██ ██ ██   ██ ██    ██ ████   ██ ████  ████ ██      ████   ██    ██
# █████   ██ ██  ██ ██    ██ ██ ██████  ██    ██ ██ ██  ██ ██ ████ ██ █████   ██ ██  ██    ██
# ██      ██  ██ ██  ██  ██  ██ ██   ██ ██    ██ ██  ██ ██ ██  ██  ██ ██      ██  ██ ██    ██
# ███████ ██   ████   ████   ██ ██   ██  ██████  ██   ████ ██      ██ ███████ ██   ████    ██

IMAGE_SIZE = env.read("IMAGE_SIZE", int, 128)  # was 512; 128 is the highest power of 2 that fits the data
BATCH_SIZE = env.read("BATCH_SIZE", int, 32)
NUM_CLASSES = 2
DIR_RAINFALLWATER = env.read("DIR_RAINFALLWATER", str)
PATH_HEIGHTMAP = env.read("PATH_HEIGHTMAP", str)
PATH_COLOURMAP = env.read("PATH_COLOURMAP", str)
PARALLEL_READS = env.read("PARALLEL_READS", float, 1.5)
STEPS_PER_EPOCH = env.read("STEPS_PER_EPOCH", int, None)
REMOVE_ISOLATED_PIXELS = env.read("NO_REMOVE_ISOLATED_PIXELS", bool, True) # This will flip True → False so it's okay
EPOCHS = env.read("EPOCHS", int, 25)
LOSS = env.read("LOSS", str, "cross-entropy-dice")  # other possible values: cross-entropy, mean-squared-error (bleh)
DICE_LOG_COSH = env.read("DICE_LOG_COSH", bool, False)
LEARNING_RATE = env.read("LEARNING_RATE", float, 0.00001)
WATER_THRESHOLD = env.read("WATER_THRESHOLD", float, 0.1, do_none=True)
UPSAMPLE = env.read("UPSAMPLE", int, 2)
SPLIT_VALIDATE = env.read("SPLIT_VALIDATE", float, 0.2)
SPLIT_TEST = env.read("SPLIT_TEST", float, 0)
# NOTE: RANDSEED is declared and handled in src/lib/dataset/primitives/shuffle.py

STEPS_PER_EXECUTION = env.read("STEPS_PER_EXECUTION", int, 1)
JIT_COMPILE = env.read("JIT_COMPILE", bool, True)
DIR_OUTPUT = env.read("DIR_OUTPUT", str, f"output/{datetime.utcnow().date().isoformat()}_deeplabv3plus_rainfall_TEST")
PATH_CHECKPOINT = env.read("PATH_CHECKPOINT", str, None)
PREDICT_COUNT = env.read("PREDICT_COUNT", int, 25)
PREDICT_AS_ONE = env.read("PREDICT_AS_ONE", bool, False)

# ~~~

env.val_dir_exists(os.path.join(DIR_OUTPUT, "checkpoints"), create=True)

# ~~~

logger.info("DeepLabV3+ rainfall radar TEST")


if LOSS == "mean-squared-error":
	logger.warning(
		"Regression mode detected via LOSS==mean-squared-error, changing NUM_CLASSES=2→1 and WATER_THRESHOLD=None"
	)
	# This is required because DeepLabV3+ requires that it format the output as N=value classes
	NUM_CLASSES = 1
	WATER_THRESHOLD = None


env.print_all(False)



# ██████   █████  ████████  █████  ███████ ███████ ████████ 
# ██   ██ ██   ██    ██    ██   ██ ██      ██         ██    
# ██   ██ ███████    ██    ███████ ███████ █████      ██    
# ██   ██ ██   ██    ██    ██   ██      ██ ██         ██    
# ██████  ██   ██    ██    ██   ██ ███████ ███████    ██    

if not PREDICT_AS_ONE:
	dataset_train, dataset_validate, dataset_test = dataset_mono(
		dirpath_input=DIR_RAINFALLWATER,
		batch_size=BATCH_SIZE,
		water_threshold=WATER_THRESHOLD,
		rainfall_scale_up=2, # done BEFORE cropping to the below size
		output_size=IMAGE_SIZE,
		input_size="same",
		filepath_heightmap=PATH_HEIGHTMAP,
		do_remove_isolated_pixels=REMOVE_ISOLATED_PIXELS,
		parallel_reads_multiplier=PARALLEL_READS,
		percentage_validate=SPLIT_VALIDATE,
		percentage_test=SPLIT_TEST
	)

	logger.info("Train Dataset:", dataset_train)
	logger.info("Validation Dataset:", dataset_validate)
	logger.info("Test Dataset:", dataset_test)
else:
	dataset_train = dataset_mono_predict(
		dirpath_input=DIR_RAINFALLWATER,
		batch_size=BATCH_SIZE,
		water_threshold=WATER_THRESHOLD,
		rainfall_scale_up=2,  # done BEFORE cropping to the below size
		output_size=IMAGE_SIZE,
		input_size="same",
		filepath_heightmap=PATH_HEIGHTMAP,
		do_remove_isolated_pixels=REMOVE_ISOLATED_PIXELS
	)
	logger.info("Dataset AS_ONE:", dataset_train)

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
	
	
	def DeeplabV3Plus(image_size, num_classes, num_channels=3, backbone="resnet", upsample=2):
		model_input = tf.keras.Input(shape=(image_size, image_size, num_channels))
		if upsample > 1:
			logger.info(f"[DeepLabV3+] Upsample enabled @ {upsample}x")
			x = tf.keras.layers.UpSampling2D(size=2)(model_input)
		else:
			logger.info("[DeepLabV3+] Upsample disabled")
			x = model_input
		
		match backbone:
			case "resnet":
				backbone = tf.keras.applications.ResNet50(
					weights="imagenet" if num_channels == 3 else None,
					include_top=False, input_tensor=x
				)
			case _:
				raise Exception(f"Error: Unknown backbone {backbone}")
		
		x = backbone.get_layer("conv4_block6_2_relu").output
		x = DilatedSpatialPyramidPooling(x)
		
		factor = 4 if upsample == 2 else 8 # else: upsample == 1. other values are not supported yet because maths
		input_a = tf.keras.layers.UpSampling2D(
			size=(image_size // factor // x.shape[1] * 2, image_size // factor // x.shape[2] * 2), # <--- UPSAMPLE after pyramid
			interpolation="bilinear",
		)(x)
		input_b = backbone.get_layer("conv2_block3_2_relu").output
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

	model = DeeplabV3Plus(
		image_size=IMAGE_SIZE,
		num_classes=NUM_CLASSES,
		upsample=UPSAMPLE,
		num_channels=8
	)
	summarywriter(model, os.path.join(DIR_OUTPUT, "summary.txt"))
else:
	model = tf.keras.models.load_model(PATH_CHECKPOINT, custom_objects={
		# Tell Tensorflow about our custom layers so that it can deserialise models that use them
		"LossCrossEntropyDice": LossCrossEntropyDice,
		"metric_dice_coefficient": dice_coefficient,
		"sensitivity": sensitivity,
		"specificity": specificity,
		"one_hot_mean_iou": mean_iou
	})


# ████████ ██████   █████  ██ ███    ██ ██ ███    ██  ██████  
#    ██    ██   ██ ██   ██ ██ ████   ██ ██ ████   ██ ██       
#    ██    ██████  ███████ ██ ██ ██  ██ ██ ██ ██  ██ ██   ███ 
#    ██    ██   ██ ██   ██ ██ ██  ██ ██ ██ ██  ██ ██ ██    ██ 
#    ██    ██   ██ ██   ██ ██ ██   ████ ██ ██   ████  ██████  

def plot_metric(train, val, name, dir_output):
	plt.plot(train, label=f"train_{name}")
	plt.plot(val, label=f"val_{name}")
	plt.title(name)
	plt.xlabel("epoch")
	plt.ylabel(name)
	plt.savefig(os.path.join(dir_output, f"{name}.png"))
	plt.close()

if PATH_CHECKPOINT is None:
	loss_fn = None
	metrics = [
		"accuracy",
		dice_coefficient,
		mean_iou(),
		sensitivity(),  # How many true positives were accurately predicted
		specificity,  # How many true negatives were accurately predicted?
	]
	if LOSS == "cross-entropy-dice":
		loss_fn = LossCrossEntropyDice(log_cosh=DICE_LOG_COSH)
	elif LOSS == "cross-entropy":
		loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
	elif LOSS == "mean-squared-error":
		loss_fn = (
			tf.keras.losses.MeanSquaredError()
		)  # TODO consider L2 loss as it might be more computationally efficient?
		metrics = [
			tf.keras.metrics.MeanSquaredError(),
			tf.keras.metrics.RootMeanSquaredError(),
			tf.keras.metrics.MeanAbsoluteError(),
		]  # Others don't make sense w/o this - NOTE this is mse and not rmse!
	else:
		raise Exception(
			f"Error: Unknown loss function '{LOSS}' (possible values: cross-entropy, cross-entropy-dice)."
		)

	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
		loss=loss_fn,
		metrics=metrics,
		steps_per_execution=STEPS_PER_EXECUTION,
		jit_compile=JIT_COMPILE,
	)
	logger.info(">>> Beginning training")
	history = model.fit(
		dataset_train,
		validation_data=dataset_validate,
		# test_data=dataset_test, # Nope, it doesn't have a param like this so it's time to do this the *hard* way
		epochs=EPOCHS,
		callbacks=[
			CallbackExtraValidation(
				{  # `model,` removed 'ref apparently exists by default????? ehhhh...???
					"test": dataset_test  # Can be None because it handles that
				}
			),
			tf.keras.callbacks.CSVLogger(
				filename=os.path.join(DIR_OUTPUT, "metrics.tsv"), separator="\t"
			),
			CallbackCustomModelCheckpoint(
				model_to_checkpoint=model,
				filepath=os.path.join(
					DIR_OUTPUT,
					"checkpoints",
					"checkpoint_e{epoch:d}_loss{loss:.3f}.hdf5",
				),
				monitor="loss",
			),
		],
		steps_per_epoch=STEPS_PER_EPOCH,
		# use_multiprocessing=True #  commented out but could be a good idea to squash warning? alt increase batch size..... but that uses more memory >_<
	)
	logger.info(">>> Training complete")
	logger.info(">>> Plotting graphs")

	plot_metric(
		history.history["loss"],
		history.history["val_loss"],
		"loss",
		DIR_OUTPUT
	)
	if WATER_THRESHOLD is None:
		plot_metric(
			history.history["mean_absolute_error"],
			history.history["val_mean_absolute_error"],
			"mean absolute error",
			DIR_OUTPUT,
		)
		plot_metric(
			history.history["mean_squared_error"],
			history.history["val_mean_squared_error"],
			"mean squared error",
			DIR_OUTPUT,
		)
		plot_metric(
			history.history["root_mean_squared_error"],
			history.history["val_root_mean_squared_error"],
			"root mean squared error",
			DIR_OUTPUT,
		)
	else:
		plot_metric(
			history.history["accuracy"],
			history.history["val_accuracy"],
			"accuracy",
			DIR_OUTPUT,
		)
		plot_metric(
			history.history["metric_dice_coefficient"],
			history.history["val_metric_dice_coefficient"],
			"dice",
			DIR_OUTPUT,
		)
		plot_metric(
			history.history["one_hot_mean_iou"],
			history.history["val_one_hot_mean_iou"],
			"mean iou",
			DIR_OUTPUT,
		)
		plot_metric(
			history.history["sensitivity"],
			history.history["val_sensitivity"],
			"sensitivity",
			DIR_OUTPUT,
		)
		plot_metric(
			history.history["specificity"],
			history.history["val_specificity"],
			"specificity",
			DIR_OUTPUT,
		)
	

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


def infer(model, image_tensor, do_argmax=True):
	# we tf.expand_dims here to add a batch size to the input to the model
	predictions = model.predict(tf.expand_dims((image_tensor), axis=0))
	predictions = tf.squeeze(predictions) # Remove the batch size dimension again
	return predictions


def decode_segmentation_masks(mask, colormap, n_classes):
	r = np.zeros_like(mask).astype(np.uint8)
	g = np.zeros_like(mask).astype(np.uint8)
	b = np.zeros_like(mask).astype(np.uint8)
	for label_index in range(0, n_classes):
		idx = mask == label_index
		r[idx] = colormap[label_index, 0]
		g[idx] = colormap[label_index, 1]
		b[idx] = colormap[label_index, 2]
	rgb = np.stack([r, g, b], axis=2)
	return rgb


def get_overlay(image, coloured_mask):
	global cv2
	# import cv2 only when used, since it might not be available and this function isn't currently used (was prob something I dropped halfway through writing 'cause I got distracted)
	if not cv2:
		cv2 = __import__("cv2") 
	image = tf.keras.preprocessing.image.array_to_img(image)
	image = np.array(image).astype(np.uint8)
	overlay = cv2.addWeighted(image, 0.35, coloured_mask, 0.65, 0)
	return overlay


def plot_samples_matplotlib(filepath, display_list):
	plt.figure(figsize=(16, 8))
	for i in range(len(display_list)):
		plt.subplot(2, math.ceil(len(display_list) / 2), i+1)
		if display_list[i].shape[-1] == 3:
			plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
		else:
			plt.imshow(display_list[i])
			plt.colorbar()
	plt.savefig(filepath, dpi=200)

def save_samples(filepath, save_list):
	handle = io.open(filepath, "a")
	json.dump(save_list, handle)
	handle.write("\n")
	handle.close()

def plot_predictions(filepath_output, input_items, colormap, model):
	filepath_jsonl = filepath_output.replace("_$$", "").replace(".png", ".jsonl")
	if os.path.exists(filepath_jsonl):
		os.truncate(filepath_jsonl, 0)
	
	i = 0
	for input_pair in input_items:
		prediction_mask = infer(image_tensor=input_pair[0], model=model)
		prediction_mask_argmax = tf.argmax(prediction_mask, axis=2)
		# label_colourmap = decode_segmentation_masks(input_pair[1], colormap, 2)
		prediction_colormap = decode_segmentation_masks(prediction_mask_argmax, colormap, 2)
		
		# print("DEBUG:plot_predictions INFER", str(prediction_mask.numpy().tolist()).replace("], [", "],\n["))
		
		plot_samples_matplotlib(
			filepath_output.replace("$$", str(i)),
			[
				# input_tensor,
				tf.math.reduce_max(input_pair[0][:,:,:-1], axis=-1), # rainfall only
				input_pair[0][:,:,-1], # heightmap
				input_pair[1], #label_colourmap,
				prediction_mask[:,:,1],
				prediction_colormap
			]
		)
		
		save_samples(
			filepath_jsonl,
			prediction_mask.numpy().tolist()
		)
		i += 1

def plot_predictions_regressive(filepath_output, input_items, model):
    # Iterate over items
    i = 0
    for input_pair in input_items:
        item_in = input_pair[0]
        item_label = input_pair[1]

        # Pre-emptive logging
        print("DEBUG:plot_predictions_regressive item_in.shape", item_in.shape)
        print("DEBUG:plot_predictions_regressive item_label.shape", item_label.shape)

        # item_in.shape (128, 128, 8)
        # item_label.shape (128, 128, 1)

        item_in = tf.expand_dims(item_in, axis=0)  # Now (1, 128, 128, 8) to add batch size

        # TODO we probably need to rework some shapes here
        prediction = model.predict(item_in)

        print("DEBUG:plot_predictions_regressive prediction.shape", prediction.shape)

        plot_samples_matplotlib(
            filepath_output.replace("$$", str(i)),
            [
                tf.math.reduce_max(input_pair[0][:, :, :-1], axis=-1),  # rainfall only
                item_label,  # absolute output
                prediction,  # prediction as image
            ],
        )

        i += 1


def plot_predictions_switcher(filepath_output, input_items, colormap, model):
	"""Switches dynamically between the regressive and classification plotting backends.

	Args:
		filepath_output (string): The filepath to output things to. $$ isr eplaced with the index in input_items that is being processed.
		input_items (list<[Tensor, Tensor]>): List of items to process. Each item in the list should be in the form [ tensor_input, tensor_label ].
		colormap (ColorMap): The colour map thingy. I forget the specifics of where this came from.
		model (tf.keras.Model): The model to use to make predictions.
	
	Returns:
		any: The return value from the relevant backend, which is usually void/None.
	"""
	if WATER_THRESHOLD is None:
		return plot_predictions_regressive(filepath_output, input_items, model)
	else:
		return plot_predictions(filepath_output, input_items, colormap, model)

def get_from_batched(dataset, count):
	result = []
	for batched in dataset:
		items_input = tf.unstack(batched[0], axis=0)
		items_label = tf.unstack(batched[1], axis=0)
		for item in zip(items_input, items_label):
			result.append(item)
			if len(result) >= count:
				return result

plot_predictions_switcher(
	os.path.join(DIR_OUTPUT, "predict_train_$$.png"),
	get_from_batched(dataset_train, PREDICT_COUNT),
	colormap,
	model=model
)
if not PREDICT_AS_ONE:
	plot_predictions_switcher(
		os.path.join(DIR_OUTPUT, "predict_validate_$$.png"),
		get_from_batched(dataset_validate, PREDICT_COUNT),
		colormap,
		model=model
	)
	if dataset_test is not None:
		plot_predictions_switcher(
			os.path.join(DIR_OUTPUT, "predict_test_$$.png"),
			get_from_batched(dataset_test, PREDICT_COUNT),
			colormap,
			model=model
		)

logger.info(f"Complete at {str(datetime.now().isoformat())}, elapsed {str((datetime.now() - time_start).total_seconds())} seconds")
