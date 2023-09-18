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
links:
  - title: GitHub
    description: GitHub is the world's largest software development platform.
    website: https://github.com
    image: https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png
---

*Coming soon !*

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
    in_size: int,
    out_size: int,
    random_key: random.PRNGKey,
    scale: float = 1e-3,
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
    return jnp.array(random.normal(w_key, (out_size, in_size))) * scale, jnp.array(
        random.normal(b_key, (out_size,)) * scale
    )


def init_mlp(
    layer_sizes: list,
    key: random.PRNGKey,
) -> list[tuple[jnp.array, jnp.array]]:
    """
    Initialize the weights and biases for a multi-layer perceptron (MLP).

    Args:
        layer_sizes (list): A list of integers representing the number of neurons in each layer,
                            where the first element is the input size, and the last element is the output size.
        key (random.PRNGKey): A random key for parameter initialization.

    Returns:
        list: A list of tuples, each containing initialized weights and biases for each layer.
    """
    keys = random.split(key, len(layer_sizes))
    return [
        init_weights_biases(in_size, out_size, key)
        for in_size, out_size, key in zip(layer_sizes[:-1], layer_sizes[1:], keys)
    ]
```

```python

def relu(x) -> jnp.array:
    """
    Rectified Linear Unit (ReLU) activation function.

    Args:
        x: Input data.

    Returns:
        jnp.array: Output after applying ReLU activation.
    """
    return jnp.maximum(0, x)

@partial(vmap, in_axes=(None, 0))
def forward(
    params,
    image,
) -> jnp.array:
    """
    Forward pass of a multi-layer perceptron (MLP).

    Args:
        params: List of tuples containing weights and biases for each layer.
        image: Input data.

    Returns:
        jnp.array: The output logits of the MLP after the final layer.
    """
    activations = image
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = relu(outputs)
    final_w, final_b = params[-1]
    logits = jnp.dot(final_w, activations) + final_b
    return logits - logsumexp(logits)
```

``` python
def one_hot(x, n_classes=n_classes) -> jnp.array:
    """
    Convert integer labels to one-hot encoding.

    Args:
        x: Integer labels.
        n_classes: Total number of classes.

    Returns:
        jnp.array: One-hot encoded labels.
    """
    return jnp.array(x[:, None] == jnp.arange(n_classes), dtype=jnp.float32)
```

``` python
def loss(params, images, targets) -> jnp.array:
    """
    Compute the categorical cross-entropy loss between predicted and target values.

    Args:
        params: List of tuples containing weights and biases for each layer.
        images: Batch of input data.
        targets: Target one-hot encoded labels.

    Returns:
        jnp.array: Batched categorical cross-entropy loss.
    """
    preds = forward(params, images)
    return -jnp.mean(preds * targets)  # cateogorical cross entropy
```

``` python
def accuracy(params, images, targets) -> jnp.array:
    """
    Compute the accuracy of predictions given the true labels.

    Args:
        params: List of tuples containing weights and biases for each layer.
        images: Input data.
        targets: Target one-hot encoded labels.

    Returns:
        jnp.array: Accuracy of predictions.
    """
    predictions = forward(params, images)
    predicted_class = jnp.argmax(predictions, axis=1)
    target_class = jnp.argmax(targets, axis=1)
    return jnp.mean(predicted_class == target_class)
```

``` python
@jit
def update(params, x, y) -> list[tuple[jnp.array, jnp.array]]:
    """
    Update the model parameters using gradient descent.

    Args:
        params: List of tuples containing weights and biases for each layer.
        x: Input data.
        y: Target one-hot encoded labels.

    Returns:
        list[tuple[jnp.array, jnp.array]: Updated model parameters after one step of gradient descent.
    """
    grads = grad(loss)(params, x, y)
    return [
        (w - learning_rate * dw, b - learning_rate * db)
        for (w, b), (dw, db) in zip(params, grads)
    ]
```

``` python
def train_loop(
    params,
    train_images,
    test_images,
    train_labels,
    test_labels,
    n_epochs=10,
) -> jnp.array:
    """
    Training loop

    Args:
        params: Model parameters.
        train_images: Training data images.
        test_images: Test data images.
        train_labels: Training data labels.
        test_labels: Test data labels.
        n_epochs: Number of training epochs.

    Returns:
        jnp.array: The fitted weights and biases
    """
    for epoch in tqdm(range(n_epochs)):
        start_time = time.time()
        for x, y in training_generator:
            y = one_hot(y, n_classes)
            params = update(params, x, y)
        epoch_time = time.time() - start_time

        train_acc = accuracy(params, train_images, train_labels)
        test_acc = accuracy(params, test_images, test_labels)

        print(f"Epoch {epoch +1} in {epoch_time:.2f} seconds")
        print(f"Training accuracy: {train_acc}")
        print(f"Test accuracy: {test_acc}")
    return params
```

```python
if __name__ == "__main__":
    params = init_mlp(layer_sizes, random_key)

    mnist_dataset = MNIST(
        "jax_ml/data/mnist", download=True, transform=FlattenAndCast()
    )
    mnist_dataset_test = MNIST("jax_ml/data/mnist", download=True, train=False)

    training_generator = NumpyLoader(
        mnist_dataset, batch_size=batch_size, num_workers=0
    )

    train_images = np.array(mnist_dataset.train_data).reshape(
        len(mnist_dataset.train_data), -1
    )
    train_labels = one_hot(np.array(mnist_dataset.train_labels), n_classes)

    test_images = jnp.array(
        mnist_dataset_test.test_data.numpy().reshape(
            len(mnist_dataset_test.test_data), -1
        ),
        dtype=jnp.float32,
    )
    test_labels = one_hot(np.array(mnist_dataset_test.test_labels), n_classes)

    train_loop(
        params=params,
        train_images=train_images,
        test_images=test_images,
        train_labels=train_labels,
        test_labels=test_labels,
        n_epochs=n_epochs,
    )
```
