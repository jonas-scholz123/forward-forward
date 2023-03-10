from torch.utils import data
import jax.numpy as jnp
import numpy as np

class NumpyLoader(data.DataLoader):
  def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
    super(self.__class__, self).__init__(dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=numpy_collate,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn)

def numpy_collate(batch):
  if isinstance(batch[0], np.ndarray):
    return np.stack(batch)
  elif isinstance(batch[0], (tuple,list)):
    transposed = zip(*batch)
    return [numpy_collate(samples) for samples in transposed]
  else:
    return np.array(batch)

class FlattenAndCast(object):
  def __call__(self, pic):
    return np.ravel(np.array(pic, dtype=jnp.float32))

def prepend_labels(xs, ys):
  # Don't modify original data.
  xs = xs.copy()

  # Set the y'th pixel to max.
  xs[np.arange(len(xs)), ys] = xs.max()
  return xs


# TODO: This is kinda slow, better way?
def randint_except(low, high, exc):
  choices = [el for el in range(low, high) if el != exc]
  return np.random.choice(choices)

def prepend_false_labels(xs, ys):
  # Don't modify original data.
  xs = xs.copy()
  not_ys = [randint_except(0, 9, y) for y in ys]

  # Set the y'th pixel to max.
  xs[np.arange(len(xs)), not_ys] = xs.max()
  return xs