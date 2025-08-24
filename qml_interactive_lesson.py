
# -*- coding: utf-8 -*-
"""
Quantum Machine Learning (QML) Interactive Lesson

This script provides an introduction to the exciting field of Quantum Machine
Learning. We'll explore the basic concepts and build a simple quantum classifier.

Author: Gemini
Date: 2025-08-24
"""

# #############################################################################
# # Part 1: Introduction to Quantum Machine Learning                       #
# #############################################################################

# Welcome to the world of Quantum Machine Learning! QML is an emerging field
# that combines the principles of quantum mechanics and machine learning. It has
# the potential to revolutionize how we solve complex problems.

# ---
# Key QML Concepts You'll Encounter:
# ---
# 1.  **Qubit:** A qubit is the basic unit of quantum information. Unlike a
#     classical bit, which can be either 0 or 1, a qubit can be in a
#     superposition of both states.
# 2.  **Quantum Gates:** These are the building blocks of quantum circuits. They
#     are used to manipulate the state of qubits.
# 3.  **Quantum Circuit:** A quantum circuit is a sequence of quantum gates that
#     performs a specific computation.
# 4.  **Variational Quantum Classifier:** This is a type of QML model that uses a
#     parameterized quantum circuit to classify data.

# #############################################################################
# # Part 2: Setup and Data Generation                                      #
# #############################################################################

import pennylane as qml
from pennylane import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# ---
# Data Generation
# ---
# We'll use a simple dataset of two interleaving half-circles (moons) to train
# our quantum classifier.

X, y = make_moons(n_samples=100, noise=0.1, random_state=42)
y = y * 2 - 1  # Rescale labels to -1 and 1

# #############################################################################
# # Part 3: Building the Quantum Classifier                                #
# #############################################################################

# ---
# Quantum Device
# ---
# We need a quantum device to run our circuit. PennyLane provides a variety of
# simulators and also supports real quantum hardware.

dev = qml.device("default.qubit", wires=2)

# ---
# Quantum Circuit
# ---
# This is the heart of our QML model. We'll create a parameterized quantum
# circuit that can be trained to classify our data.

@qml.qnode(dev)
def quantum_circuit(params, x):
    qml.RY(x[0], wires=0)
    qml.RY(x[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(params[0], wires=0)
    qml.RY(params[1], wires=1)
    return qml.expval(qml.PauliZ(0))

# ---
# The Classifier
# ---
# We'll create a classifier that uses the quantum circuit to make predictions.

def classifier(params, x):
    return quantum_circuit(params, x)

# #############################################################################
# # Part 4: Training the Model                                             #
# #############################################################################

# ---
# Cost Function
# ---
# We need a cost function to measure how well our model is performing.

def cost(params, X, y):
    predictions = [classifier(params, x) for x in X]
    return np.mean((predictions - y) ** 2)

# ---
# Training
# ---
# We'll use an optimizer to find the best parameters for our circuit.

params = np.random.uniform(0, np.pi, 2)
optimizer = qml.AdamOptimizer(stepsize=0.1)

for i in range(100):
    params, cost_val = optimizer.step_and_cost(lambda p: cost(p, X, y), params)
    if (i + 1) % 10 == 0:
        print(f"Step {i+1:3d}: Cost = {cost_val:.4f}")

# #############################################################################
# # Part 5: Visualizing the Results                                        #
# #############################################################################

# Let's visualize the decision boundary of our trained classifier.

xx, yy = np.meshgrid(np.linspace(-1.5, 2.5, 50), np.linspace(-1, 1.5, 50))
Z = np.array([classifier(params, [x, y]) for x, y in zip(xx.ravel(), yy.ravel())])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, levels=[-1, 0, 1], cmap="RdBu", alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu", edgecolors="k")
plt.title("Decision Boundary of the Quantum Classifier")
plt.show()

# ---
# Expert QML Researcher's Thought:
# ---
# "This is a simple example, but it demonstrates the basic principles of QML.
# The real power of QML will come from its ability to solve problems that are
# intractable for classical computers. As quantum hardware continues to improve,
# we can expect to see more and more applications of QML in various fields."
