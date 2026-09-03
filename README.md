# From-Scratch Neural Network & Autograd Framework

A lightweight neural network and automatic differentiation framework implemented from scratch in Python and NumPy.

This project was developed progressively to explore how modern deep learning frameworks work internally. Starting from a basic feed-forward neural network system, the framework was gradually extended to support convolutional networks, recurrent networks, normalization layers, and residual connections.

The goal of the project is not to replace existing libraries such as PyTorch or TensorFlow, but to build a simplified deep learning framework from the ground up and gain a better understanding of model construction, forward propagation, backpropagation, parameter optimization, and reusable neural network abstractions.

---

## Implemented Features

The framework currently supports several major neural network architectures and components:

- Automatic differentiation and backpropagation
- Dense / feed-forward neural networks
- Common activation and loss functions
- SGD-based parameter optimization
- Sequential model construction
- Convolutional neural networks
- Pooling and feature-flattening layers
- Softmax and multiclass classification
- Recurrent neural networks
- Sequence processing and backpropagation through time
- Layer normalization and batch normalization
- Additive and concatenative skip connections

The project was developed cumulatively across four stages:

1. **Core Autograd & Feed-Forward Networks**
2. **Convolutional Neural Network Extension**
3. **Recurrent Neural Network Extension**
4. **Normalization & Residual Connection Extension**

Each stage builds directly on the previous implementation, gradually expanding the same underlying framework rather than creating separate standalone models.

---

## Project Structure

The framework follows a modular design where neural network components can be combined to construct different architectures.

At a high level, the project includes:

- Neural network layers
- Activation functions
- Loss functions
- Optimizers
- Trainable parameters
- Sequential models
- Convolutional modules
- Recurrent modules
- Normalization layers
- Skip connections

This modular structure allows existing components to be reused when implementing more complex neural network architectures.

---

## Project Goals

This project focuses on understanding the internal mechanics behind deep learning frameworks, including:

- how neural network layers are represented as reusable modules;
- how forward and backward passes interact;
- how gradients propagate through different architectures;
- how trainable parameters are managed and updated;
- and how increasingly complex neural network components can be built on top of a shared framework.

---

## Future Improvements

The current framework can be further expanded in several directions.

Possible future additions include:

- **LSTM and GRU cells** for more advanced recurrent architectures
- **Adam, RMSProp, and other optimizers**
- **Dropout and additional regularization techniques**
- **Learnable parameters for normalization layers**
- **Additional convolution and pooling variants**
- **More flexible model composition beyond sequential architectures**
- **Attention mechanisms**
- **Transformer components**
- **GPU acceleration**
- **Improved vectorization and computational performance**
- **Model serialization and checkpointing**
- **Expanded unit tests and gradient checking utilities**

Longer term, the framework could evolve into a small general-purpose deep learning library capable of constructing and training a wider range of neural network architectures.

---

## Tech-Stacks

- Python
- NumPy
- Object-Oriented Programming
- Neural Networks
- Automatic Differentiation

---

## Status

The core framework currently supports feed-forward, convolutional, recurrent, normalization, and residual neural network components.

Further development will focus on improving performance, expanding supported architectures, and adding additional training and usability features.

The modular design makes it possible to extend the framework with additional components such as:

- LSTM cells
- GRU cells
- Dropout
- additional optimizers such as Adam
- learnable normalization parameters
- additional convolution and pooling variants
- additional residual architectures
- attention mechanisms
- Transformer components
...