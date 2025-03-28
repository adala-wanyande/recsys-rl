# Assignment 1: Neural Collaborative Filtering (NCF)

Link to official instructions: https://brightspace.universiteitleiden.nl/d2l/lms/dropbox/user/folder_submit_files.d2l?db=186378&grpid=387754&isprv=0&bp=0&ou=331576

## Objective

In this assignment, you will implement Neural Collaborative Filtering (NCF) to predict user-item interactions. NCF is a powerful recommendation model that combines matrix factorization with deep neural networks. Your goal is to:

- Implement an NCF model using PyTorch.
- Train the model on the MovieLens 1M dataset.
- Evaluate performance using Recall@10 and NDCG@10.

You may refer to the official NCF GitHub repository:  
https://github.com/hexiangnan/neural_collaborative_filtering

---

## Dataset

**MovieLens 1M Dataset**  
Download from the [official MovieLens website](https://grouplens.org/datasets/movielens/1m/).

### Format:
userId::movieId::rating::timestamp
### Preprocessing Instructions:

- Treat ratings **≥ 4** as **positive interactions** (`label = 1`).
- Generate **implicit negative interactions** (`label = 0`) via negative sampling from movies the user has not interacted with.

---

## Implementation Requirements

You are expected to implement the following components:

### 1. Data Preprocessing (20 points)
- Load and preprocess the MovieLens dataset.
- Convert explicit ratings to binary implicit feedback.
- Implement negative sampling (e.g., randomly sample non-interacted items as negatives).
- Randomly split the dataset:
  - 70% for training
  - 15% for validation
  - 15% for testing

### 2. Neural Collaborative Filtering Model (30 points)
Implement an NCF model that includes:
- Embedding layers for users and items.
- A GMF (Generalized Matrix Factorization) branch.
- An MLP (Multi-Layer Perceptron) branch.
- A fusion layer to combine GMF and MLP outputs.
- A final prediction layer using a sigmoid activation.

Implementation should be done in **PyTorch**.

### 3. Training and Optimization (20 points)
- Implement the training loop using **binary cross-entropy loss**.
- Use the **Adam** optimizer.
- Implement **early stopping** based on validation loss.

### 4. Evaluation (15 points)
- Evaluate model performance using the following metrics:
  - **Recall@10**
  - **NDCG@10**
- Compare results across different model settings (e.g., varying MLP layers).

### 5. Experiment Report (15 points)
Write a 2–3 page concise report in **LaTeX** format using this [Springer LNCS Template](https://www.overleaf.com/latex/templates/ants2024-template-based-on-springer-lcns/fmhprnymcbpr), including:
- Description of dataset preprocessing
- Model architecture details (GMF, MLP, fusion strategy)
- Training details and hyperparameter choices
- Evaluation results and comparisons
- Key observations and ideas for improvement

---

## Submission Requirements

Each student must submit the following:

- Source code files (`.py`) with detailed comments.
- Experiment report (`.pdf`) written in LaTeX format.
- A `README.md` file explaining how to run your code.

---

## Grading Breakdown (100 points)

| Task                         | Points | Description                                                             |
|------------------------------|--------|-------------------------------------------------------------------------|
| Data Preprocessing           | 20     | Proper loading, binary feedback conversion, and negative sampling       |
| NCF Model Implementation     | 30     | Embeddings, GMF/MLP branches, fusion, sigmoid output                    |
| Training & Optimization      | 20     | Loss, optimizer, early stopping                                         |
| Evaluation                   | 15     | Recall@10 and NDCG@10 implementation and comparison                     |
| Experiment Report            | 15     | LaTeX report with clear methodology and insights                        |
| **Total**                    | **100**|                                                                         |

---

## Deadline

- **Submission Deadline**: 6 April 2025, 23:59 (via Brightspace)
- **Late Submission Penalty**: 5% per day
- **No Credit**: Submissions made after 13 April 2025, 23:59

> Note: There is no retake option for the assignment, as an additional one-week grace period is already included. If you submit after April 13 or fail the assignment (e.g., non-functioning code), you will be required to retake the course next academic year.

---

## Support

For questions, post on the discussion forum or contact the course instructor.
