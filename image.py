import matplotlib.pyplot as plt
import tensorflow as tf
from IPython.display import clear_output

def display(image):
	plt.figure(figsize=(15, 15))
	plt.imshow(tf.keras.utils.array_to_img(image))
	plt.axis('off')
	plt.show()