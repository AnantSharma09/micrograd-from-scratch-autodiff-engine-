# Micrograd From Scratch

A complete implementation of an automatic differentiation engine and neural network library from scratch in Python, inspired by Andrej Karpathy's Micrograd. This project demonstrates how reverse-mode automatic differentiation powers neural network training without relying on deep learning frameworks such as PyTorch or TensorFlow.

## Features

- Dynamic computational graph
- Reverse-mode automatic differentiation
- Automatic gradient accumulation
- Topological graph traversal for backpropagation
- Operator overloading
- Fully connected neural networks
- Gradient descent optimization

## Project Structure

```
micrograd-from-scratch/
├── engine.py      # Automatic differentiation engine
├── nn.py          # Neural network modules (Neuron, Layer, MLP)
├── train.py       # Example training script
├── README.md
├── requirements.txt
└── .gitignore
```

## Example

Run the training script:

```bash
python train.py
```

## Concepts Implemented

- Value class
- Dynamic computational graph
- Reverse-mode backpropagation
- Gradient accumulation
- Topological sorting
- Operator overloading
- Neuron
- Layer
- Multi-Layer Perceptron (MLP)
- Parameter collection
- Zeroing gradients
- Gradient descent optimization
