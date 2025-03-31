import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict

# Load the data into a DataFrame
columns = ['userId', 'movieId', 'rating', 'timestamp']
df = pd.read_csv('..data/ratings.dat',
                 sep='::', names=columns, engine='python')

# Keep ratings >= 4 as implicit positive interactions
df_positive = df[df['rating'] >= 4].copy()
df_positive['interaction'] = 1  # positive interactions labeled as 1

# Drop unnecessary columns
df_positive = df_positive[['userId', 'movieId', 'interaction']]


def generate_negative_samples(df_positive, num_negatives=4, seed=42):
    np.random.seed(seed)
    user_item_set = set(zip(df_positive['userId'], df_positive['movieId']))
    all_movie_ids = set(df_positive['movieId'].unique())
    negative_samples = []

    grouped = df_positive.groupby('userId')['movieId'].apply(set).to_dict()

    for user, pos_items in grouped.items():
        num_pos = len(pos_items)
        num_to_sample = num_pos * num_negatives
        possible_neg_items = list(all_movie_ids - pos_items)

        # If not enough negatives, sample with replacement
        if len(possible_neg_items) >= num_to_sample:
            sampled_negatives = np.random.choice(
                possible_neg_items, size=num_to_sample, replace=False)
        else:
            sampled_negatives = np.random.choice(
                possible_neg_items, size=num_to_sample, replace=True)

        for item in sampled_negatives:
            negative_samples.append([user, item, 0])  # label as negative

    df_negative = pd.DataFrame(negative_samples, columns=[
                               'userId', 'movieId', 'interaction'])
    return df_negative


# Generate negatives (4 negatives per positive interaction)
df_negative = generate_negative_samples(df_positive, num_negatives=4)


# Positive interactions
df_positive_final = df_positive[['userId', 'movieId', 'interaction']]

# Combine data
df_full = pd.concat([df_positive_final, df_negative], ignore_index=True)

# Shuffle data
df_full = df_full.sample(frac=1, random_state=42).reset_index(drop=True)


# First split: 70% training, 30% temp
train_df, temp_df = train_test_split(
    df_full, test_size=0.30, random_state=42, stratify=df_full['interaction'])

# Second split: 50% of temp (15% validation, 15% testing)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=42, stratify=temp_df['interaction'])

print(f"Train size: {len(train_df)} ({len(train_df)/len(df_full)*100:.2f}%)")
print(f"Validation size: {len(val_df)} ({len(val_df)/len(df_full)*100:.2f}%)")
print(f"Test size: {len(test_df)} ({len(test_df)/len(df_full)*100:.2f}%)")

train_df.to_csv('..data/train_df.csv', index=False)
val_df.to_csv('..data/val_df.csv', index=False)
test_df.to_csv('..data/test_df.csv', index=False)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get number of users and items
num_users = train_df['userId'].max() + 1
num_items = train_df['movieId'].max() + 1

# 2. Define PyTorch Dataset


class MovieLensDataset(Dataset):
    def __init__(self, df):
        self.users = df['userId'].values
        self.items = df['movieId'].values
        self.labels = df['interaction'].values

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.users[idx], dtype=torch.long),
            torch.tensor(self.items[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float)
        )


# Data loaders
batch_size = 1024
train_loader = DataLoader(MovieLensDataset(
    train_df), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(MovieLensDataset(val_df), batch_size=batch_size)
test_loader = DataLoader(MovieLensDataset(test_df), batch_size=batch_size)

# 3. Define NCF Model


class NeuralCF(nn.Module):
    def __init__(self, num_users, num_items, embed_size=32, mlp_layers=[64, 32, 16, 8]):
        super(NeuralCF, self).__init__()

        # GMF embeddings
        self.user_embedding_gmf = nn.Embedding(num_users, embed_size)
        self.item_embedding_gmf = nn.Embedding(num_items, embed_size)

        # MLP embeddings
        self.user_embedding_mlp = nn.Embedding(num_users, embed_size)
        self.item_embedding_mlp = nn.Embedding(num_items, embed_size)

        # MLP layers
        mlp_modules = []
        input_size = embed_size * 2
        for layer_size in mlp_layers:
            mlp_modules.append(nn.Linear(input_size, layer_size))
            mlp_modules.append(nn.ReLU())
            input_size = layer_size
        self.mlp = nn.Sequential(*mlp_modules)

        # Fusion Layer (combine GMF + MLP)
        fusion_input_size = embed_size + mlp_layers[-1]
        self.fusion_layer = nn.Linear(fusion_input_size, 1)

        # Final sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, user, item):
        # GMF branch
        user_gmf = self.user_embedding_gmf(user)
        item_gmf = self.item_embedding_gmf(item)
        gmf_output = user_gmf * item_gmf  # Element-wise multiplication

        # MLP branch
        user_mlp = self.user_embedding_mlp(user)
        item_mlp = self.item_embedding_mlp(item)
        mlp_input = torch.cat([user_mlp, item_mlp], dim=-1)
        mlp_output = self.mlp(mlp_input)

        # Fusion
        fusion_input = torch.cat([gmf_output, mlp_output], dim=-1)
        logits = self.fusion_layer(fusion_input)
        output = self.sigmoid(logits)

        return output.squeeze()


# Instantiate model
model = NeuralCF(num_users, num_items).to(device)

# 4. Loss and Optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 5. Training Function


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for users, items, labels in loader:
        users = users.to(device)
        items = items.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(users, items)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)

# 6. Validation Function


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for users, items, labels in loader:
            users = users.to(device)
            items = items.to(device)
            labels = labels.to(device)

            outputs = model(users, items)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(loader)


# Early stopping parameters
patience = 3  # Number of epochs to wait before stopping
best_val_loss = np.inf
epochs_without_improvement = 0
num_epochs = 20  # maximum epochs, adjust as needed

for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss = evaluate(model, val_loader, criterion)

    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    # Check if validation loss improved
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        # Save the best model
        torch.save(model.state_dict(), '..models/ncf_model_best.pt')
        print("Validation loss improved, saving model.")
    else:
        epochs_without_improvement += 1
        print(f"No improvement for {epochs_without_improvement} epoch(s).")

    # Stop training if no improvement for 'patience' epochs
    if epochs_without_improvement >= patience:
        print(f"Early stopping triggered after {epoch+1} epochs.")
        break

# Load best model weights after training
model.load_state_dict(torch.load('..models/ncf_model_best.pt'))

# 7. Evaluation on Test Set
model.eval()
# Create a set of all items in the training set for negative sampling
all_items = test_df['movieId'].unique()
train_user_item_set = set(zip(train_df['userId'], train_df['movieId']))

# Generate evaluation set: for each positive test instance, generate 9 negative samples


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
