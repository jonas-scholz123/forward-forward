import matplotlib.pyplot as plt
import numpy as np

def show(inputs, idx=0):
  im = inputs[idx].reshape(28, 28)
  plt.figure(figsize = (4, 4))
  plt.imshow(im, cmap="gray")
  plt.show()

def batch(arr, batch_size):
  for i in range(0, len(arr), batch_size):
    yield arr[i: i+batch_size]