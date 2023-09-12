---
title: "Building a Neural Network with JAX: An Illustrated Guide"
description: Let's code a Multi-Layer Perceptron from scratch with JAX !
slug: jax_mlp
date: 2023-09-12 00:00:00+0000
image: cover.jpg
categories:
    - JAX
    - Machine Learning
tags: 
    - Implementation
weight: 1       # You can add weight to some posts to override the default sorting (date descending)
math: true
---

```python
import jax.numpy as jnp
from jax import random, tree_map, vmap
from jax.scipy.special import logsumexp
```

```python
LAYER_SIZES = [784, 512, 512, 10]
LEARNING_RATE = 1e-2
N_EPOCHS = 8
BATCH_SIZE = 128
N_CLASSES = 10
RANDOM_SEED = 0
```

```python
def init_weights_biases(
    in_size: int, out_size: int, random_key: random.PRNGKey, scale: float = 1e-3
) -> tuple[jnp.array, jnp.array]:
    """
    Initialize weights and biases for a neural network layer.

    Args:
        in_size (int): Number of input features.
        out_size (int): Number of output neurons.
        random_key (random.PRNGKey): Random key for reproducible random numbers.
        scale (float, optional): Scaling factor for weight initialization. Defaults to 1e-3.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: A tuple containing initialized weights and biases.
    """
    w_key, b_key = random.split(random_key)
    return scale * random.normal(w_key, (out_size, in_size)), scale * random.normal(
        b_key, (out_size,)
    )


def random_layer_init(layer_sizes: list, random_key: random.PRNGKey) -> list[tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Initialize weights and biases for all layers in a neural network.

    Args:
        layer_sizes (List[int]): List of layer sizes, including input and output dimensions.
        random_key (random.PRNGKey): Random key for reproducible random numbers.

    Returns:
        List[Tuple[jnp.ndarray, jnp.ndarray]]: A list of tuples, each containing initialized weights and biases for a layer.
    """
    keys = random.split(random_key, len(layer_sizes))
    return [
        init_weights_biases(in_size, out_size, key)
        for in_size, out_size, key in zip(layer_sizes[:-1], layer_sizes[1:], keys)
    ]
```

```python

def relu(x):
    """ """
    return jnp.maximum(0, x)

def predict(params, image):
    """ """
    # initialize the activations
    activations = image
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = relu(outputs)
    
    final_w, final_b = params[-1]
    logits = jnp.dot(final_w, activations) + final_b

    return logits - logsumexp(logits)
```

```python
def loss(params: list, images: jnp.array, targets: jnp.array) -> float:
    preds = batched_predict(params, images)
    return -jnp.mean(preds * targets)

```

```python
if __name__ == "__main__":
    key = random.PRNGKey(RANDOM_SEED)
    params = random_layer_init(LAYER_SIZES, key)

    print(tree_map(lambda x: x.shape, params))

    batched_predict = vmap(predict, in_axes=(None, 0))
    random_flattened_image = random.normal(
        random.PRNGKey(1),
        (
            10,
            28 * 28,
        ),
    )
    preds = batched_predict(params, random_flattened_image)
    print(preds.shape)

```
