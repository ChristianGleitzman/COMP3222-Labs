# Lab 6: the perceptron

The exercises are in `Lab6.pdf` in the repository root. Run this from the repository root:

```bash
python code/lab6/perceptron.py
```

This is a **student starter**. The data, `plot_boundary`, `add_bias`, `perceptron_predict`, `ACTIVATIONS`, `neuron_fit` and `neuron_predict` are provided. The lines marked `# TODO` are yours: the three lines in `perceptron_fit` (Part 1) and `sigmoid`, `tanh` and `relu` (Part 3). The `__main__` block lists the steps for each part as comments.

| Part | What it covers |
|---|---|
| 1 | Train a perceptron on logical AND, and see how the learning rate and starting weights change the epochs needed |
| 2 | Two Gaussian classes with scikit-learn's `Perceptron` and with your own, and whether a line can separate them |
| 3 | Sigmoid, tanh and ReLU neurons trained by gradient descent, compared on time and accuracy |
| Optional | Breast cancer, with and without `StandardScaler` |
