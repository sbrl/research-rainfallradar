#!/usr/bin/env python3
# @source https://keras.io/examples/vision/deeplabv3_plus/
# Required dataset: https://drive.google.com/uc?id=1B9A9UCJYMwTL4oBEo4RZfbMZMaZhKJaz [instance-level-human-parsing.zip]

from datetime import datetime
from loguru import logger
from lib.ai.helpers.summarywriter import summarywriter

import os
import cv2
import numpy as np
from glob import glob
from scipy.io import loadmat
import matplotlib.pyplot as plt

import tensorflow as tf

from lib.dataset.dataset_mono import dataset_mono

IMAGE_SIZE = int(os.environ["IMAGE_SIZE"]) if "IMAGE_SIZE" in os.environ else 128 # was 512; 128 is the highest power of 2 that fits the data
BATCH_SIZE = int(os.environ["BATCH_SIZE"]) if "BATCH_SIZE" in os.environ else 64
NUM_CLASSES = 2
DIR_RAINFALLWATER = os.environ["DIR_RAINFALLWATER"]
PATH_HEIGHTMAP = os.environ["PATH_HEIGHTMAP"]
PATH_COLOURMAP = os.environ["PATH_COLOURMAP"]
STEPS_PER_EPOCH = int(os.environ["STEPS_PER_EPOCH"]) if "STEPS_PER_EPOCH" in os.environ else None

DIR_OUTPUT=os.environ["DIR_OUTPUT"] if "DIR_OUTPUT" in os.environ else f"output/{datetime.utcnow().date().isoformat()}_deeplabv3plus_rainfall_TEST"

if not os.path.exists(DIR_OUTPUT):
	os.makedirs(DIR_OUTPUT)

logger.info("DeepLabV3+ rainfall radar TEST")
logger.info(f"> BATCH_SIZE {BATCH_SIZE}")
logger.info(f"> DIR_RAINFALLWATER {DIR_RAINFALLWATER}")
logger.info(f"> PATH_HEIGHTMAP {PATH_HEIGHTMAP}")
logger.info(f"> PATH_COLOURMAP {PATH_COLOURMAP}")
logger.info(f"> STEPS_PER_EPOCH {STEPS_PER_EPOCH}")
logger.info(f"> DIR_OUTPUT {DIR_OUTPUT}")


dataset_train, dataset_validate = dataset_mono(
	dirpath_input=DIR_RAINFALLWATER,
	batch_size=BATCH_SIZE,
	water_threshold=0.1,
	rainfall_scale_up=2, # done BEFORE cropping to the below size
	output_size=IMAGE_SIZE,
	input_size="same",
	filepath_heightmap=PATH_HEIGHTMAP,
)

logger.info("Train Dataset:", dataset_train)
logger.info("Validation Dataset:", dataset_validate)


# ███    ███  ██████  ██████  ███████ ██     
# ████  ████ ██    ██ ██   ██ ██      ██     
# ██ ████ ██ ██    ██ ██   ██ █████   ██     
# ██  ██  ██ ██    ██ ██   ██ ██      ██     
# ██      ██  ██████  ██████  ███████ ███████

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
    resnet50 = tf.keras.applications.ResNet50(
        weights="imagenet" if num_channels == 3 else None,
		include_top=False, input_tensor=model_input
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    input_a = tf.keras.layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = tf.keras.layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = tf.keras.layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = tf.keras.layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
    return tf.keras.Model(inputs=model_input, outputs=model_output)


model = DeeplabV3Plus(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES, num_channels=8)
summarywriter(model, os.path.join(DIR_OUTPUT, "summary.txt"))




# ████████ ██████   █████  ██ ███    ██ ██ ███    ██  ██████  
#    ██    ██   ██ ██   ██ ██ ████   ██ ██ ████   ██ ██       
#    ██    ██████  ███████ ██ ██ ██  ██ ██ ██ ██  ██ ██   ███ 
#    ██    ██   ██ ██   ██ ██ ██  ██ ██ ██ ██  ██ ██ ██    ██ 
#    ██    ██   ██ ██   ██ ██ ██   ████ ██ ██   ████  ██████  

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=loss,
    metrics=["accuracy"],
)
logger.info(">>> Beginning training")
history = model.fit(dataset_train,
	validation_data=dataset_validate,
	epochs=25,
	callbacks=[
		tf.keras.callbacks.CSVLogger(
			filename=os.path.join(DIR_OUTPUT, "metrics.tsv"),
			separator="\t"
		)
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


def plot_samples_matplotlib(filepath, display_list, figsize=(5, 3)):
    _, axes = plt.subplots(nrows=1, ncols=len(display_list), figsize=figsize)
    for i in range(len(display_list)):
        if display_list[i].shape[-1] == 3:
            axes[i].imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        else:
            axes[i].imshow(display_list[i])
    plt.savefig(filepath)


def plot_predictions(filepath, input_items, colormap, model):
    for input_tensor in input_items:
        prediction_mask = infer(image_tensor=input_tensor, model=model)
        prediction_colormap = decode_segmentation_masks(prediction_mask, colormap, 20)
        overlay = get_overlay(input_tensor, prediction_colormap)
        plot_samples_matplotlib(
			filepath,
            [input_tensor, overlay, prediction_colormap],
			figsize=(18, 14)
        )

def get_items_from_batched(dataset, count):
	result = []
	for batched in dataset:
		items = tf.unstack(batched, axis=0)
		for item in items:
			result.append(item)
			if len(result) >= count:
				return result


plot_predictions(
	os.path.join(DIR_OUTPUT, "predict_train.png"),
	get_items_from_batched(dataset_train, 4),
	colormap,
	model=model
)
plot_predictions(
	os.path.join(DIR_OUTPUT, "predict_validate.png"),
	get_items_from_batched(dataset_validate, 4),
	colormap,
	model=model
)
