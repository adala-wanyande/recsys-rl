import pandas as pd
import numpy as np
import torch
from collections import defaultdict


# Assume you've loaded your trained NCF model already.
import torch
from CNF import NeuralCF  # Import model class clearly from NCF.py

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define parameters exactly as used during training
num_users = 6041  # Replace this with the exact number from your training script
num_items = 3953  # Replace this with the exact number from your training script
embed_size = 32
mlp_layers = [64, 32, 16, 8]

# Initialize your model clearly (must match the saved model exactly)
model = NeuralCF(num_users, num_items, embed_size, mlp_layers).to(device)

# Load trained weights clearly
model.load_state_dict(torch.load(
    'models/ncf_model_best.pt', map_location=device))

# Set model to evaluation mode
model.eval()

print("✅ NCF Model loaded and ready for evaluation.")

# Load test data
test_df = pd.read_csv('data/test_df.csv')
train_df = pd.read_csv('data/train_df.csv')

all_items = test_df['movieId'].unique()
train_user_item_set = set(zip(train_df['userId'], train_df['movieId']))

# Generate evaluation set: for each positive test instance, generate 99 negative samples


def get_eval_data(test_df, num_negatives=9):
    eval_users, eval_items, eval_labels = [], [], []

    for row in test_df.itertuples():
        u = row.userId
        pos_item = row.movieId

        neg_items = []
        while len(neg_items) < num_negatives:
            neg_item = np.random.choice(all_items)
            if (u, neg_item) not in train_user_item_set and neg_item != pos_item:
                neg_items.append(neg_item)

        # Positive instance
        eval_users.append(u)
        eval_items.append(pos_item)
        eval_labels.append(1)

        # Negative instances
        for neg_item in neg_items:
            eval_users.append(u)
            eval_items.append(neg_item)
            eval_labels.append(0)

    return torch.tensor(eval_users), torch.tensor(eval_items), torch.tensor(eval_labels)


eval_users, eval_items, eval_labels = get_eval_data(test_df)


def recall_at_k(actual, predicted, k=10):
    actual_set = set(actual)
    predicted = predicted[:k]
    hits = len([item for item in predicted if item in actual_set])
    return hits / len(actual_set)


def ndcg_at_k(actual, predicted, k=10):
    dcg = 0.0
    for i, p in enumerate(predicted[:k]):
        if p in actual:
            dcg += 1 / np.log2(i + 2)
    idcg = sum([1 / np.log2(i + 2) for i in range(min(len(actual), k))])
    return dcg / idcg if idcg > 0 else 0.0


# Group evaluation data by user
user_eval_dict = defaultdict(list)
for user, item, label in zip(eval_users, eval_items, eval_labels):
    user_eval_dict[user.item()].append((item.item(), label.item()))

recall_scores = []
ndcg_scores = []

with torch.no_grad():
    for user, item_label_list in user_eval_dict.items():
        items = torch.tensor([x[0] for x in item_label_list]).to(device)
        labels = [x[1] for x in item_label_list]

        user_tensor = torch.tensor([user]*len(items)).to(device)

        predictions = model(user_tensor, items)
        predictions = predictions.cpu().numpy()

        # Rank items by predicted scores
        ranked_items = [x for _, x in sorted(
            zip(predictions, items.cpu().numpy()), reverse=True)]

        actual_items = [item for item, label in zip(
            items.cpu().numpy(), labels) if label == 1]

        recall = recall_at_k(actual_items, ranked_items, k=10)
        ndcg = ndcg_at_k(actual_items, ranked_items, k=10)

        recall_scores.append(recall)
        ndcg_scores.append(ndcg)

print(f"Recall@10: {np.mean(recall_scores):.4f}")
print(f"NDCG@10: {np.mean(ndcg_scores):.4f}")
