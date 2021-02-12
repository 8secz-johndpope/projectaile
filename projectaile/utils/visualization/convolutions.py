import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def visualize_activations(model, image):
    imgs_per_row = 16
    outputs = [layer.output for layer in model.layers[1:]]
    names = [layer.name for layer in model.layers[1:]]
    visualization_model = tf.keras.models.Model(inputs=model.inputs, outputs=outputs)
    activations = visualization_model.predict(image)
    for name, activation in zip(names, activations):
        n_features = activation.shape[-1] # Number of features in the feature map
        size = activation.shape[1] #The feature map has shape (1, size, size, n_features).
        n_cols = n_features // imgs_per_row # Tiles the activation channels in this matrix
        display_grid = np.zeros((size * n_cols, imgs_per_row * size))
        for col in range(n_cols): # Tiles each filter into a big horizontal grid
            for row in range(imgs_per_row):
                channel_image = activation[0,:, :, col * imgs_per_row + row]*255
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto')
        

def saliency_maps():
    return