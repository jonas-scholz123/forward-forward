#%%
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm 

import jax
import jax.numpy as jnp
from jax import jit, vmap, value_and_grad, random
import optax

from typing import NamedTuple, Tuple, List
import functools
import pickle

from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader

class MNIST_Dataset(Dataset):
    def __init__(self, train):
        super().__init__()

        self.train = train
        self.base_dataset = MNIST(root='train_mnist', download=True, train=train)

    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, index: int):
        img = self.base_dataset[index][0]
        label = self.base_dataset[index][1]

        # TODO: Normalize here
        img = np.ravel(np.array(img, dtype=np.float32)) # Flatten Image
        img[:10] = 0
        
        if self.train:
            # Create a positive sample
            pos_img = img.copy()
            pos_img[label] = img.max()

            # Create a negative sample
            neg_img = img.copy()
            random_label = np.random.choice(np.setdiff1d(list(range(0, 10)), [label]))
            neg_img[random_label] = img.max()

            return img, pos_img, neg_img, label
        else:
            return img, label

def train_collate_fn(batch):
    transposed_data = list(zip(*batch))
    imgs = jnp.array(transposed_data[0])
    pos_imgs = jnp.array(transposed_data[1])
    neg_imgs = jnp.array(transposed_data[2])
    true_labels = jnp.array(transposed_data[3])

    return imgs, pos_imgs, neg_imgs, true_labels

def test_collate_fn(batch):
    transposed_data = list(zip(*batch))
    imgs = jnp.array(transposed_data[0])
    true_labels = jnp.array(transposed_data[1])

    return imgs, true_labels

mnist_train = MNIST_Dataset(train=True)
mnist_test = MNIST_Dataset(train=False)

train_loader = DataLoader(mnist_train, len(mnist_train), shuffle=True, collate_fn=train_collate_fn, drop_last=True)
test_loader = DataLoader(mnist_test, len(mnist_test), shuffle=False, collate_fn=test_collate_fn, drop_last=True)

class LayerState(NamedTuple):
    params: Tuple[jnp.ndarray, jnp.ndarray]
    opt_state: Tuple


def init_layer_state(in_dim, out_dim, PRNGkey, opt, scale=1e-3) -> LayerState:
    initializer = jax.nn.initializers.glorot_uniform()

    weights_key, bias_key = jax.random.split(PRNGkey)
    weights = initializer(weights_key, (out_dim, in_dim), jnp.float32) * scale
    biases = jax.random.normal(bias_key, (out_dim, ), jnp.float32) * scale
    opt_state = opt.init((weights, biases))

    return LayerState((weights, biases), opt_state)

'''
Stateless layer namespace that contains the logic for training layers.
'''
class Layer:
    def forward(params, x):
        """
        For a single flattened image x of shape (784, )
        """ 
        # Layer Normalization
        # mean = jnp.mean(x)
        # variance = jnp.var(x)
        # inv = jax.lax.rsqrt(variance + 1e-6)
        # x = (x - mean) * inv

        #x /= (jnp.linalg.norm(x, 2) + 1e-6)
        x /= (jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True)) + 1e-6)

        w, b = params
        return activation_fn(jnp.dot(w, x) + b)
        #return jax.nn.relu(jnp.dot(w, x) + b)

    @functools.partial(jax.vmap, in_axes=(None, 0)) # Batch dim 0
    def b_forward(params, xs):
        return Layer.forward(params, xs)
    
    def goodness(params, x):
        """
        Sum squared goodness for a single image - (784, )
        """

        h = Layer.forward(params, x)  
        return jnp.power(h, 2).sum(), h

    @functools.partial(jax.vmap, in_axes=(None, 0)) # Batch dim 0
    def b_goodness(params, xs):
        return Layer.goodness(params, xs)

    # TODO: variable threshold.
    def forward_forward(params, pos_xs, neg_xs):
        """
        pos_xs : batch of positive input images - (batch_size, 784)
        neg_xs : batch of negative input images - (batch_size, 784)
        """

        pos_goodness, _ = Layer.b_goodness(params, pos_xs)
        neg_goodness, _ = Layer.b_goodness(params, neg_xs)

        pos_logits = pos_goodness - pos_xs.shape[-1]
        neg_logits = neg_goodness - neg_xs.shape[-1]
        
        loss = -jax.nn.log_sigmoid(jnp.concatenate([pos_logits, -neg_logits])).mean()
        
        return loss

# Utility Functions

def plot_image(inputs, idx=0):
    print("Embedded Label:",  np.argmax(inputs[idx][:10]))
    im = inputs[idx].reshape(28, 28)
    plt.figure(figsize = (4, 4))
    plt.imshow(im, cmap="gray")
    plt.show()

def batch(arr, batch_size):
  for i in range(0, len(arr), batch_size):
    yield arr[i: i+batch_size]


def train_layer(
        opt: optax.GradientTransformation,
        num_epochs: int,
        batch_size: int,
        initial_state: LayerState,
        layer_idx: int,
        pos_xs: np.ndarray,
        neg_xs: np.ndarray):
    
    #forward_fn = functools.partial(Layer.forward_forward, )
    @jit
    def step(state: LayerState, pos_xs, neg_xs):
        loss, grads = value_and_grad(Layer.forward_forward)(state.params, pos_xs, neg_xs)
        updates, opt_state = opt.update(grads, state.opt_state)
        params = optax.apply_updates(state.params, updates)
        return loss, LayerState(params, opt_state)

    loss_hist = []

    pos_batches = list(batch(pos_xs, batch_size))
    neg_batches = list(batch(neg_xs, batch_size))

    state = initial_state

    pbar = tqdm(range(num_epochs), desc=f"Training Layer {layer_idx}")
    for epoch in pbar:
        loss_sum = 0.
        for pos_batch, neg_batch in zip(pos_batches, neg_batches):
            loss, state = step(state, pos_batch, neg_batch)
            loss_sum += loss
        loss_hist.append(float(loss_sum))
        pbar.set_postfix({'loss': loss_sum})
    
    return state, loss_hist

def train_net(
    opt: optax.GradientTransformation,
    num_epochs: int,
    batch_size: int,
    initial_states: List[LayerState],
    pos_xs: np.ndarray,
    neg_xs: np.ndarray):

    final_states = []
    loss_hists = []

    for layer_idx, layer_state in enumerate(initial_states):
        final_state, loss_hist = train_layer(opt, num_epochs, batch_size, layer_state, layer_idx, pos_xs, neg_xs)
        final_states.append(final_state)
        loss_hists.append(loss_hist)
        pos_xs = Layer.b_forward(final_state.params, pos_xs)
        neg_xs = Layer.b_forward(final_state.params, neg_xs)
    
    return final_states, loss_hists

def predict_one_sample(params, x):
    goodnesses = jnp.zeros(10)
    for query_label in range(10):
        img = jnp.array(x.copy())
        img = img.at[query_label].set(img.max())
        for layer_params in params:
            g, img = Layer.goodness(layer_params, img)
            goodnesses = goodnesses.at[query_label].add(g)

    return jnp.argmax(goodnesses)

def accuracy(params, xs, ys):
    b_predict = vmap(predict_one_sample, in_axes=(None, 0))
    y_hats = b_predict(params, xs)

    if ys.shape[-1] == 10:
        labels = jnp.argmax(ys, axis=-1)
    else:
        labels = ys

    return jnp.mean(labels == y_hats)
#%%


activation_fn = jax.nn.relu
lr = 0.001
rng = random.PRNGKey(39)
#opt = optax.adam(lr)
opt = optax.sgd(lr)
n_episodes = 60
batch_size = 100

states = [
    init_layer_state(784, 512, rng, opt, scale=1),
    init_layer_state(512, 512, rng, opt, scale=1),
    init_layer_state(512, 512, rng, opt, scale=1),
]

train_xs, train_pos_xs, train_neg_xs, train_labels = next(iter(train_loader))
states, loss_hists = train_net(opt, n_episodes, batch_size, states, train_pos_xs, train_neg_xs)
#%%

test_xs, test_labels = next(iter(test_loader))
test_acc = accuracy([s.params for s in states], test_xs, test_labels)
print("Test Accuracy:", test_acc)
#%%
plot_image(test_xs[19])
#%%
idx = 5
plot_image(test_xs, idx)
predict_one_sample([s.params for s in states], test_xs[idx])
#%%
plot_image(states[0].params[0], 11)
#%%

activation_fns = [
    jax.nn.relu,
    jax.nn.sigmoid,
    jax.nn.tanh,
    ]
names = ["relu", "sigmoid", "tanh"]
n_episodes = 60
batch_size = 128

# 2 episodes
episode_lrs = [0.01]
#episode_lrs = [0.1]

for activation_fn, name in zip(activation_fns, names):
    train_xs, train_pos_xs, train_neg_xs, train_labels = next(iter(train_loader))
    test_xs, test_labels = next(iter(test_loader))

    print("Train xs shape: ", train_pos_xs.shape)
    print("Test x, y shape: ", test_xs.shape, test_labels.shape)

    rng = random.PRNGKey(39)

    opt = optax.adam(episode_lrs[0])
    states = [
        init_layer_state(784, 512, rng, opt, scale=1),
        init_layer_state(512, 512, rng, opt, scale=1),
    ]

    for episode, lr in enumerate(episode_lrs, 1):
        print(f"Starting episode {episode}, lr = {lr}")
        opt = optax.adam(lr)
        states, loss_hists = train_net(opt, n_episodes, batch_size, states, train_pos_xs, train_neg_xs)

        for lh in loss_hists:
            plt.plot(lh)
        plt.show()
    

    train_acc = accuracy([s.params for s in states], train_xs, train_labels)
    print("Train Accuracy:", train_acc)

    train_acc_pos = accuracy([s.params for s in states], train_pos_xs, train_labels)
    print("Train Accuracy Pos:", train_acc_pos)

    test_acc = accuracy([s.params for s in states], test_xs, test_labels)
    print("Test Accuracy:", test_acc)

    results = {
        "name": name,
        "n_episodes": n_episodes,
        "states": states,
        "loss_hists": loss_hists,
        "train_acc": train_acc,
        "test_acc": test_acc,
    }

    with open(f"./results/{name}.pkl", "wb") as f:
        pickle.dump(results, f)
#%%

loss_hists

weights = jnp.zeros((500, 784)) + 2
biases = jnp.zeros(500) + 2
x = jnp.zeros(784) + 1
Layer.forward((weights, biases), x)
#%%
plt.plot(loss_hists[0])