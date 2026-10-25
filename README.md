# Machine-learning-Assignment-2

## 1. Overview

This project investigates ensemble learning for face classification using the Olivetti Faces dataset. It compares a single Decision Tree with Bagging and Boosting and evaluates model performance and robustness when part of a face image is masked.

The models use raw flattened pixel values without PCA or other dimensionality-reduction techniques.

## 2. Dataset

The dataset is loaded using `sklearn.datasets.fetch_olivetti_faces`.

- Number of identities: 40
- Images per identity: 10
- Image dimensions: 64 × 64 pixels
- Total images: 400
- Features per flattened image: 4,096

## 3. Dataset Split

The dataset is split separately within each identity.

| Set | Images per identity | Total images |
|---|---:|---:|
| Training | 6 | 240 |
| Validation | 2 | 80 |
| Testing | 2 | 80 |

**Random seed:** `26005`

The split indices are stored in `olivetti_split_seed_26005.npz`. Reuse the same split for every experiment to ensure fair model comparisons.

Load the dataset in its original order:

```python
import numpy as np
from sklearn.datasets import fetch_olivetti_faces

data = fetch_olivetti_faces(shuffle=False)
split = np.load("olivetti_split_seed_26005.npz")

train_idx = split["train_idx"]
val_idx = split["val_idx"]
test_idx = split["test_idx"]

X_train, y_train = data.data[train_idx], data.target[train_idx]
X_val, y_val = data.data[val_idx], data.target[val_idx]
X_test, y_test = data.data[test_idx], data.target[test_idx]
```

## 4. Experiments

### 4.1 Single Decision Tree

A single Decision Tree serves as the baseline classifier. Model complexity can be investigated by varying parameters such as maximum depth and minimum samples per leaf.

### 4.2 Bagging and Boosting

Bagging and Boosting are compared using the ensemble sizes specified in the assignment, including 25, 50, and 100 learners where applicable.

The evaluation includes training, validation, and clean-test accuracy, as well as training time.

- **Bagging:** Trains multiple learners on bootstrap samples and combines their predictions.
- **Boosting:** Trains learners sequentially, placing greater emphasis on examples that earlier learners classify incorrectly.

### 4.3 Robustness Evaluation

To evaluate robustness, the central 16 × 16 region of each test image is set to zero.

Using Python's zero-based indexing, the masked region is:

```python
masked_images[:, 24:40, 24:40] = 0
```

Models are trained using clean training images. Masking is applied during evaluation only.

The evaluation compares clean-test and masked-test accuracy. Additional analysis may include winning vote or probability and examples where Bagging and Boosting produce different predictions.

## 5. Requirements

Typical dependencies include:

- Python 3
- NumPy
- scikit-learn
- Matplotlib

Install them using:

```bash
pip install numpy scikit-learn matplotlib
```

## 6. How to Run

1. Clone or download the repository.
2. Install the required Python packages.
3. Place `olivetti_split_seed_26005.npz` where the experiment scripts expect it.
4. Run the Q1, Q2, and Q3 scripts using their actual filenames.
5. Reuse the same saved split for all experiments.
6. Inspect the generated logs, CSV/JSON files, and plots.
7. Regenerate the report PDF from the LaTeX source if required.

## 7. Reproducibility

- Use random seed `26005` for the dataset split.
- Reuse the saved training, validation, and test indices.
- Train models using clean training images.
- Apply masking only during robustness evaluation.
- Save model configurations, logs, and figures.
- Verify that the report's numerical results match the experiment outputs.
- Include the GitHub repository URL and final commit hash in the submitted report.

## 8. References

1. scikit-learn, [Olivetti Faces dataset documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_olivetti_faces.html).
2. scikit-learn, [Official documentation](https://scikit-learn.org/stable/).

