#%%
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib

from main import LayerState, plot_image
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


results = {}
results_path = "./results"
for sub_path in os.listdir(results_path):
    path = f"{results_path}/{sub_path}"

    name = sub_path.split(".")[0]

    with open(path, "rb") as f:
        results[name] = pickle.load(f)

#%%
names = [
    "norm_power_1",
    "norm_power_2",
    "norm_power_3",
    "norm_power_4",
]

test_accuracies = [float(results[name]["test_acc"]) for name in names]

plt.figure(figsize=(6, 2.5))
pows = [1, 2, 3, 4]
plt.bar(pows, test_accuracies, fill=False, label="Accuracy")
plt.xticks(pows)
plt.xlabel("$p$")
plt.ylabel("Accuracy using $g = ||\mathbf{h}||_p$")
plt.hlines(0.1, 0, 5, linestyles="--", color="r", label="Random Choices")
plt.xlim(0.3, 4.8)
plt.errorbar(pows, test_accuracies, [0.01, 0.02, 0.03, 0.02], fmt="+", capsize=3)
plt.legend(loc="center left")
plt.savefig("./figures/norm.png", bbox_inches="tight", dpi=500)
plt.show()

#%%
fig, ax = plt.subplots(1, 2, figsize=(6, 3), sharey=True)

names = [
    "relu",
    "sigmoid",
    "tanh"
]
labels = [
    "ReLU",
    "$\sigma$",
    "tanh"
]

test_accuracies = [float(results[name]["test_acc"]) for name in names]

plt.subplots_adjust(wspace=0.1)

ax[0].set_ylabel("Accuracy")
ax[0].bar(labels, test_accuracies, fill=True, label="Accuracy")
ax[0].set_xlabel("Activation Function")

names = [
    "relu",
    "sigmoid_loss"
]
labels = [
    "log-$\sigma$",
    "$\sigma$"
]

test_accuracies = [float(results[name]["test_acc"]) for name in names]


ax[1].bar(labels, test_accuracies, fill=True, label="Accuracy")
ax[1].set_xlabel("Loss Function")
#ax[1].errorbar([0, 1], test_accuracies, [0.01, 0.01], fmt="+", capsize=3)
plt.savefig("./figures/losses_and_activations.png", bbox_inches="tight", dpi=500)
plt.show()

#%%
nx = 5
ny = 3
fig, ax = plt.subplots(ny, nx, figsize=(nx, ny))

indices = [1, 5, 12, 37, 39, 46, 48, 57, 77, 106, 148, 223, 213, 204, 203, 202, 200, 196, 189]
bad_indices = [0, 247, 243, 235, 230, 229]

for j in range(nx):
    for i in range(ny):
        if i == ny -1:
            idx = bad_indices[j]
        else:
            idx = indices[nx * i + j]
        features = results["relu"]["states"][0].params[0]
        plot_image(features, idx, ax=ax[i, j])
        ax[i, j].set_xticks([])
        ax[i, j].set_yticks([])
        ax[i, j].set_aspect("equal")
fig.subplots_adjust(wspace=0, hspace=0)
plt.savefig("./figures/filterbank.png", dpi=500, bbox_inches="tight")
# %%
fig, ax = plt.subplots(1, 1)
plot_image(features, indices[6], ax=ax)
ax.set_yticks([])
ax.set_xticks([])
plt.savefig("./figures/9_filter.png", dpi=500, bbox_inches="tight")

#%%
fig, ax = plt.subplots(1, 1)
features = results["relu"]["states"][1].params[0]
plot_image(features, 0, ax=ax)
ax.set_yticks([])
ax.set_xticks([])
#plt.savefig("./figures/3_filter.png", dpi=500, bbox_inches="tight")