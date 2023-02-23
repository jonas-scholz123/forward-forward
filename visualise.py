import matplotlib.pyplot as plt

def show(inputs, idx=0):
  im = inputs[idx].reshape(28, 28)
  plt.figure(figsize = (4, 4))
  plt.imshow(im, cmap="gray")
  plt.show()