#!/usr/bin/env python3
# @source https://keras.io/examples/vision/deeplabv3_plus/
# Required dataset: https://drive.google.com/uc?id=1B9A9UCJYMwTL4oBEo4RZfbMZMaZhKJaz [instance-level-human-parsing.zip]

from datetime import datetime
from loguru import logger

import os
import cv2
import numpy as np
from glob import glob
from scipy.io import loadmat
import matplotlib.pyplot as plt

import tensorflow as tf

IMAGE_SIZE = 512
BATCH_SIZE = 4
NUM_CLASSES = 20
DATA_DIR = "./instance-level_human_parsing/instance-level_human_parsing/Training"
NUM_TRAIN_IMAGES = 1000
NUM_VAL_IMAGES = 50

DIR_OUTPUT=f"output/{datetime.utcnow().date().isoformat()}_deeplabv3plus_TEST"

if not os.path.exists(DIR_OUTPUT):
	os.makedirs(DIR_OUTPUT)

logger.info("DeepLabv3+ TEST")
logger.info(f"> DIR_OUTPUT {DIR_OUTPUT}")


train_images = sorted(glob(os.path.join(DATA_DIR, "Images/*")))[:NUM_TRAIN_IMAGES]
train_masks = sorted(glob(os.path.join(DATA_DIR, "Category_ids/*")))[:NUM_TRAIN_IMAGES]
val_images = sorted(glob(os.path.join(DATA_DIR, "Images/*")))[
	NUM_TRAIN_IMAGES : NUM_VAL_IMAGES + NUM_TRAIN_IMAGES
]
val_masks = sorted(glob(os.path.join(DATA_DIR, "Category_ids/*")))[
	NUM_TRAIN_IMAGES : NUM_VAL_IMAGES + NUM_TRAIN_IMAGES
]


def read_image(image_path, mask=False):
	image = tf.io.read_file(image_path)
	if mask:
		image = tf.image.decode_png(image, channels=1)
		image.set_shape([None, None, 1])
		image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
	else:
		image = tf.image.decode_png(image, channels=3)
		image.set_shape([None, None, 3])
		image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
		image = image / 127.5 - 1
	return image


def load_data(image_list, mask_list):
	image = read_image(image_list)
	mask = read_image(mask_list, mask=True)
	return image, mask


def data_generator(image_list, mask_list):
	dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
	dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
	dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
	return dataset


train_dataset = data_generator(train_images, train_masks)
val_dataset = data_generator(val_images, val_masks)

logger.info("Train Dataset:", train_dataset)
logger.info("Val Dataset:", val_dataset)


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
        weights="imagenet", include_top=False, input_tensor=model_input
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


model = DeeplabV3Plus(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)
model.summary()




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
history = model.fit(train_dataset,
	validation_data=val_dataset,
	epochs=25,
	callbacks=[
		tf.keras.callbacks.CSVLogger(
			filename=os.path.join(DIR_OUTPUT, "metrics.tsv"),
			separator="\t"
		)
	],
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
    os.path.join(os.path.dirname(DATA_DIR), "human_colormap.mat")
)["colormap"]
colormap = colormap * 100
colormap = colormap.astype(np.uint8)


def infer(model, image_tensor):
    predictions = model.predict(np.expand_dims((image_tensor), axis=0))
    predictions = np.squeeze(predictions)
    predictions = np.argmax(predictions, axis=2)
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


def get_overlay(image, colored_mask):
    image = tf.keras.preprocessing.image.array_to_img(image)
    image = np.array(image).astype(np.uint8)
    overlay = cv2.addWeighted(image, 0.35, colored_mask, 0.65, 0)
    return overlay


def plot_samples_matplotlib(filepath, display_list, figsize=(5, 3)):
    _, axes = plt.subplots(nrows=1, ncols=len(display_list), figsize=figsize)
    for i in range(len(display_list)):
        if display_list[i].shape[-1] == 3:
            axes[i].imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        else:
            axes[i].imshow(display_list[i])
    plt.savefig(filepath)


def plot_predictions(filepath, images_list, colormap, model):
    for image_file in images_list:
        image_tensor = read_image(image_file)
        prediction_mask = infer(image_tensor=image_tensor, model=model)
        prediction_colormap = decode_segmentation_masks(prediction_mask, colormap, 20)
        overlay = get_overlay(image_tensor, prediction_colormap)
        plot_samples_matplotlib(
			filepath,
            [image_tensor, overlay, prediction_colormap],
			figsize=(18, 14)
        )


plot_predictions(os.path.join(DIR_OUTPUT, "predict_train.png"), train_images[:4], colormap, model=model)
plot_predictions(os.path.join(DIR_OUTPUT, "predict_validate.png"), val_images[:4], colormap, model=model)
