import matplotlib.pyplot as plt
import matplotlib.animation as animation
import keras

class DrawWeights(keras.callbacks.Callback):

    def __init__(self, figsize, layer_id=0, param_id=0, weight_slice=(slice(None), 0)):
        self.layer_id = layer_id
        self.param_id = param_id
        self.weight_slice = weight_slice
        # Initialize the figure and axis
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(1, 1, 1)

    def on_train_begin(self,logs):
        print("Start Training ...")
        self.imgs = []

    def on_batch_end(self, batch, logs):
        # Get a snapshot of the weight matrix every 5 batches
        if batch % 5 == 0:
            # Access the full weight matrix
            weights = self.model.layers[self.layer_id].get_weights()
            # Create the frame and add it to the animation
            img = self.ax.imshow(weights[0], interpolation='nearest',aspect='auto')
            self.imgs.append([img])



    def on_train_end(self,logs):
        # Once the training has ended, display the animation
        anim = animation.ArtistAnimation(self.fig, self.imgs, interval=1, blit=False)
        plt.colorbar(self.imgs[0][0])
        plt.show()

