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
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
#%%

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
    #initializer = jax.nn.initializers.normal()

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

        x /= (jnp.linalg.norm(x, norm_power) + 1e-6)

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
        return jnp.power(h, norm_power).sum(), h

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

        #jax.debug.print("goodnesses +: {}, -: {}", pos_goodness.mean(), neg_goodness.mean())
        
        # Calculates the mean loss for a batch of images
        # TODO: Change to sigmoid for separate pos and neg loss
        #loss = jnp.log(
        #        1 + 
        #        jnp.exp(jnp.concatenate([ -pos_goodness + threshold, neg_goodness - threshold]))
        #    ).mean()
        
        logits = jnp.concatenate([pos_goodness - threshold, -neg_goodness + threshold])
        loss = loss_fn(logits).mean()
        
        return loss, (pos_goodness.mean(), neg_goodness.mean())

# Utility Functions

def plot_image(inputs, idx=0, ax=None):

    im = inputs[idx].reshape(28, 28)
    if ax==None:
        print("Embedded Label:",  np.argmax(inputs[idx][:10]))
        plt.figure(figsize = (4, 4))
        plt.imshow(im, cmap="gray")
        plt.show()
    else:
        ax.imshow(im, cmap="gray")

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
        (loss, (pos_goodness, neg_goodness)), grads = value_and_grad(Layer.forward_forward, has_aux=True)(state.params, pos_xs, neg_xs)

        #jax.debug.print("grads mean: {}", jnp.absolute(grads[0]).mean())
        updates, opt_state = opt.update(grads, state.opt_state)
        params = optax.apply_updates(state.params, updates)
        return loss, LayerState(params, opt_state), pos_goodness, neg_goodness

    loss_hist = []
    g_pos_hist = []
    g_neg_hist = []

    pos_batches = list(batch(pos_xs, batch_size))
    neg_batches = list(batch(neg_xs, batch_size))

    state = initial_state

    pbar = tqdm(range(num_epochs), desc=f"Training Layer {layer_idx}")
    for epoch in pbar:
        loss_sum = 0.
        for pos_batch, neg_batch in zip(pos_batches, neg_batches):
            loss, state, g_pos, g_neg = step(state, pos_batch, neg_batch)
            g_pos_hist.append(g_pos)
            g_neg_hist.append(g_neg)
            #jax.debug.print("avg_weight {}", state.params[0].mean())
            loss_sum += loss
        loss_hist.append(float(loss_sum))
        pbar.set_postfix({'loss': loss_sum})
    
    return state, loss_hist, g_pos_hist, g_neg_hist

def train_net(
    opt: optax.GradientTransformation,
    num_epochs: int,
    batch_size: int,
    initial_states: List[LayerState],
    pos_xs: np.ndarray,
    neg_xs: np.ndarray):

    final_states = []
    loss_hists = []
    g_pos_hists = []
    g_neg_hists = []

    for layer_idx, layer_state in enumerate(initial_states):
        final_state, loss_hist, g_pos_hist, g_neg_hist = train_layer(opt, num_epochs, batch_size, layer_state, layer_idx, pos_xs, neg_xs)
        g_pos_hists.append(g_pos_hist)
        g_neg_hists.append(g_neg_hist)
        final_states.append(final_state)
        loss_hists.append(loss_hist)
        pos_xs = Layer.b_forward(final_state.params, pos_xs)
        neg_xs = Layer.b_forward(final_state.params, neg_xs)
    
    return final_states, loss_hists, g_pos_hists, g_neg_hists

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

#activation_fns = [
#    #jax.nn.sigmoid,
#    jax.nn.tanh,
#    ]
#names = [
#    "sigmoid",
#    "tanh",
#    ]
#thresholds = [1]
#n_episodes = 100
#batch_size = 1024
#
## 2 episodes
#episode_lrs = [0.1, 0.05]
##episode_lrs = [0.1]
#norm_power = 2
#
#for activation_fn, name, threshold in zip(activation_fns, names, thresholds):
#    train_xs, train_pos_xs, train_neg_xs, train_labels = next(iter(train_loader))
#    test_xs, test_labels = next(iter(test_loader))
#
#    print("Train xs shape: ", train_pos_xs.shape)
#    print("Test x, y shape: ", test_xs.shape, test_labels.shape)
#
#    rng = random.PRNGKey(39)
#
#    opt = optax.adam(episode_lrs[0])
#    states = [
#        init_layer_state(784, 512, rng, opt, scale=1e-5),
#        init_layer_state(512, 512, rng, opt, scale=1e-5),
#    ]
#
#    for episode, lr in enumerate(episode_lrs, 1):
#        print(f"Starting episode {episode}, lr = {lr}, threshold = {threshold}")
#        opt = optax.adam(lr)
#        states, loss_hists = train_net(opt, n_episodes, batch_size, states, train_pos_xs, train_neg_xs)
#
#        for lh in loss_hists:
#            plt.plot(lh)
#        plt.show()
#    
#
#    train_acc = accuracy([s.params for s in states], train_xs, train_labels)
#    print("Train Accuracy:", train_acc)
#
#    train_acc_pos = accuracy([s.params for s in states], train_pos_xs, train_labels)
#    print("Train Accuracy Pos:", train_acc_pos)
#
#    test_acc = accuracy([s.params for s in states], test_xs, test_labels)
#    print("Test Accuracy:", test_acc)
#
#    results = {
#        "name": name,
#        "n_episodes": n_episodes,
#        "states": states,
#        "loss_hists": loss_hists,
#        "train_acc": train_acc,
#        "test_acc": test_acc,
#    }
#
#    with open(f"./results/{name}.pkl", "wb") as f:
#        pickle.dump(results, f)

if __name__ == "__main__":
    activation_fn = jax.nn.relu
    threshold = 10
    n_episodes = 100
    batch_size = 128

    # 2 episodes
    min_lr = 0.01
    max_lr = 1
    lrs = np.logspace(np.log10(min_lr), np.log10(max_lr), 15)

    lrs = [0.72]

    norm_power = 2
    loss_fn = lambda x: -jax.nn.log_sigmoid(x)
    #loss_fn = lambda x: jax.nn.sigmoid(-x)

    mom_results = {}
    momentums = [0, 0.9]
    momentums = [0]
    for momentum in momentums:
        pos_results = {}
        neg_results = {}
        accuracies = {}
        for lr in lrs:
            train_xs, train_pos_xs, train_neg_xs, train_labels = next(iter(train_loader))
            test_xs, test_labels = next(iter(test_loader))

            print("Train xs shape: ", train_pos_xs.shape)
            print("Test x, y shape: ", test_xs.shape, test_labels.shape)

            rng = random.PRNGKey(39)
            states = [
                init_layer_state(784, 512, rng, opt, scale=1e-2),
                init_layer_state(784, 512, rng, opt, scale=1e-2),
            ]

            print(f"Starting, lr = {lr}")
            opt = optax.sgd(lr, momentum=momentum)
            states, loss_hists, g_pos_hists, g_neg_hists = train_net(opt, n_episodes, batch_size, states, train_pos_xs, train_neg_xs)

            train_acc = accuracy([s.params for s in states], train_xs, train_labels)
            print("Train Accuracy:", train_acc)

            train_acc_pos = accuracy([s.params for s in states], train_pos_xs, train_labels)
            print("Train Accuracy Pos:", train_acc_pos)

            test_acc = accuracy([s.params for s in states], test_xs, test_labels)
            print("Test Accuracy:", test_acc)

            accuracies[lr] = test_acc
            pos_results[lr] = g_pos_hists[0]
            neg_results[lr] = g_neg_hists[0]

            name = f"norm_power_{norm_power}"

            results = {
                "name": name,
                "n_episodes": n_episodes,
                "states": states,
                "loss_hists": loss_hists,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "episode_lrs": [lr],
            }
    
            with open(f"./results/{name}.pkl", "wb") as f:
                pickle.dump(results, f)

        mom_results[momentum] = {
            "accuracies": accuracies,
            "pos_results": pos_results,
            "neg_results": neg_results,
        }

#%%

#%%
plt.figure(figsize=(6, 3))
plt.xlabel("Learning Rate $\\alpha$")
plt.ylabel("Test Accuracy after 10 epochs")
plt.xscale("log")

momentums = [0, 0.9]
for mom in momentums:
    accuracies = mom_results[mom]["accuracies"]
    xs = list(accuracies.keys())
    ys = list(accuracies.values())
    plt.plot(xs, ys, "x-", label=f"Momentum = {mom}")
plt.legend()
plt.savefig("./figures/lr_sensitivity.pdf", bbox_inches="tight")

#%%
accuracies.items()

#%%
mom = 0.9
accs = list(mom_results[mom]["accuracies"].values())
best_idx = np.argmax(accs)
lo_idx = 5
hi_idx = -3
show_indices = [lo_idx, best_idx, hi_idx]
show_lrs = [list(mom_results[mom]["accuracies"].keys())[idx] for idx in show_indices]
pos_results = list(mom_results[mom]["pos_results"].values())

hi_lim = 5000
lo_lim = 300000

g_pos_lo = np.array(pos_results[lo_idx])[:lo_lim]
g_pos_best = np.array(pos_results[best_idx])
g_pos_hi = np.array(pos_results[hi_idx])[:hi_lim]

neg_results = list(mom_results[mom]["neg_results"].values())
g_neg_lo = np.array(neg_results[lo_idx])[:lo_lim]
g_neg_best = np.array(neg_results[best_idx])
g_neg_hi = np.array(neg_results[hi_idx])[:hi_lim]

fig, ax = plt.subplots(1, 3, figsize=(6, 3))

for i, (pos, neg) in enumerate(zip(
    [g_pos_lo, g_pos_best, g_pos_hi],
    [g_neg_lo, g_neg_best, g_neg_hi])):

    g_pos = np.array(pos)
    g_neg = np.array(neg)

    ax[i].plot(g_pos, label="$g_{pos}$")
    ax[i].plot(g_neg, label="$g_{neg}$")
    xmax = len(g_pos) * 1.1
    ax[i].set_xlim(-0.05 * xmax, xmax)
    ax[i].hlines(10, -100, 100000, linestyles="--", color="r", label="$\\theta$")
    ax[i].set_xlabel("Batch")

    label_lr = round(show_lrs[i], 2)
    ax[i].set_title(f"$\\alpha = {label_lr}$")
    #plt.ylim(0.001, 100)
#ax[0].set_yscale("log")
ax[-1].legend(loc="upper right")
plt.savefig("./figures/goodness_divergence_mom.pdf", bbox_inches="tight")
#%%
plt.plot(g_neg_best)
plt.plot(g_pos_best)