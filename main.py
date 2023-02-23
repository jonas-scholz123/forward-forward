#%%
import functools
import jax
import jax.numpy as jnp
from jax import jit, vmap, value_and_grad
from jax import random
import numpy as np
from torch.utils import data
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from functools import partial
from typing import NamedTuple, Tuple
import optax

from dataloaders import FlattenAndCast, NumpyLoader, prepend_false_labels, prepend_labels
from visualise import show

def relu(x):
    return jnp.maximum(0, x)

class LayerState(NamedTuple):
    params: Tuple[jnp.ndarray, jnp.ndarray]
    opt_state: Tuple

def init_layer_state(shape, rng, opt) -> LayerState:
    weights_key, bias_key = jax.random.split(rng)
    weights = jax.random.normal(weights_key, shape).T
    biases = jax.random.normal(bias_key, [shape[1]])
    opt_state = opt.init((weights, biases))
    return LayerState((weights, biases), opt_state)

def predict_labels(params, xs):
    goodnesses = np.zeros((10, xs.shape[0]))
    b_fw = vmap(Layer.forward, (None, 0))

    for query_label in range(9):
        h = prepend_labels(xs, query_label)

        for layer_params in params:
            #TODO: speed this up, avoid double computation
            g = Layer.b_goodness(layer_params, h)
            h = b_fw(layer_params, h)
            goodnesses[query_label] += g

    return np.argmax(goodnesses, axis=0)

def accuracy(params, xs, ys):
  y_hats = predict_labels(params, xs)

  if ys.shape[-1] == 10:
    labels = jnp.argmax(ys, axis=-1)
  else:
    labels = ys
  return jnp.mean(labels == y_hats)


'''
Stateless layer namespace that contains the logic for training layers.
'''
class Layer:
    #TODO: Make activation function a static input.
    def forward(params, x):
        # Normalise inputs, add small positive value to avoid NaNs
        x /= (jnp.linalg.norm(x, 2) + 1e-6)
        w, b = params
        return relu(w @ x + b)
    
    # Expands function to include a 0th batch-size dimension.
    @functools.partial(jax.vmap, in_axes=(None, 0))
    def b_goodness(params, x):
        h = Layer.forward(params, x)
        return jnp.power(h, 2).mean()

    #TODO: variable threshold.
    def loss(params, pos_xs, neg_xs):
        threshold = 2
        pos_goodness = Layer.b_goodness(params, pos_xs)
        neg_goodness = Layer.b_goodness(params, neg_xs)

        loss = jnp.log(1 + jnp.exp(jnp.concatenate([
                    - pos_goodness + threshold,
                    neg_goodness - threshold]))).mean()

        return loss
    
    def b_forward(params, xs):
        batched_forward = vmap(Layer.forward, (None, 0))
        return batched_forward(xs)

def train(opt: optax.GradientTransformation, num_epochs: int, initial_state: LayerState, xs, ys):
    @jit
    def step(state: LayerState, pos_xs, neg_xs):
        loss, grads = value_and_grad(Layer.loss)(state.params, pos_xs, neg_xs)
        updates, opt_state = opt.update(grads, state.opt_state)
        params = optax.apply_updates(state.params, updates)
        return loss, LayerState(params, opt_state)

    #TODO: chunk & batch
    pos_xs = prepend_labels(xs, ys)
    neg_xs = prepend_false_labels(xs, ys)
    loss_hist = []

    state = initial_state

    for _ in tqdm(range(num_epochs)):
        loss, state = step(state, pos_xs, neg_xs)
        loss_hist.append(loss)
    
    return state, loss_hist

transform = Compose([
    ToTensor(),
    Normalize((0.1307,), (0.3081,)),
    FlattenAndCast(),
    ])

mnist_train = MNIST('/tmp/mnist/', download=True, train=True, transform=transform)
training_generator = NumpyLoader(mnist_train, batch_size=60000, num_workers=0)

opt = optax.adam(0.03)
rng = random.PRNGKey(42)
state = init_layer_state((784, 512), rng, opt)

xs, ys = next(iter(training_generator))
state, loss_hist = train(opt, 30, state, xs, ys)

#%%

accuracy([state.params], xs, ys)
#%%
plt.plot(loss_hist)
a = predict_labels([state.params], xs)

#%%
for i in range(30):
  show(state.params[0], i)
