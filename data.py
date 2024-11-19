import os
import numpy as np
import keras
from keras import layers
from tensorflow import data as tf_data
import matplotlib.pyplot as plt

def delete_corrupted():
	num_skipped = 0
	for folder_name in ("Cat", "Dog"):
		folder_path = os.path.join("PetImages", folder_name)
		for fname in os.listdir(folder_path):
			fpath = os.path.join(folder_path, fname)
			try:
				fobj = open(fpath, "rb")
				is_jfif = b"JFIF" in fobj.peek(10)
			finally:
				fobj.close()

			if not is_jfif:
				num_skipped += 1
				# Delete corrupted image
				os.remove(fpath)

	print(f"Deleted {num_skipped} images.")


def generate_dataset():
	image_size = (180,180)
	batch_size= 16

	train_ds, val_ds = keras.utils.image_dataset_from_directory("PetImages", validation_split=0.2, subset="both", seed=1982, image_size=image_size, batch_size=batch_size)
	return train_ds, val_ds


data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]


def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images
