Iris Neural Network Classifier (PyTorch)

This project implements a simple **2-layer neural network classifier** from scratch using **PyTorch** to classify flowers from the famous **Iris dataset**.

---

##  Project Overview

- Dataset: **Iris** (150 samples, 3 classes: Setosa, Versicolor, Virginica)
- Framework: **PyTorch** (no high-level wrappers used)
- Model: **Two-layer fully connected neural network**
- Optimizer: **Adam**
- Loss: **CrossEntropyLoss**
- Plot: Accuracy vs Epoch (train and test)
- Output: Trained model performance and a saved plot

---

##  Requirements

Install dependencies using:

```bash
pip install torch matplotlib scikit-learn
```

Or if using a **virtual environment** (recommended):

```bash
python -m venv .venv
.venv\Scripts\activate           # For Windows
# source .venv/bin/activate      # For macOS/Linux

pip install torch matplotlib scikit-learn
```

---

##  How to Run

Once dependencies are installed, simply run the script:

```bash
python classifier.py
```

You will see output for all 50 epochs and final test accuracy like this:

```
Epoch [1/50], Loss: ..., Train Acc: ..., Test Acc: ...
...
Final Test Accuracy: 93.33%
Plot saved as accuracy_plot.png
```

---

##  Output

After training:
- Accuracy results are printed to the terminal.
- A plot is saved as:
  ```
  accuracy_plot.png
  ```
  showing both training and testing accuracy over 50 epochs.

---

##  Plot Display Notes

This script uses:

```python
matplotlib.use('Agg')
```

Which disables GUI plotting (to avoid errors on systems without `tkinter`). It saves the plot silently.

If you **want to display the plot** on systems with GUI support, you can:

1. Comment out:
   ```python
   matplotlib.use('Agg')
   ```
2. Uncomment:
   ```python
   plt.show()
   ```

---
