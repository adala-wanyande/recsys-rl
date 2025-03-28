import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Load Preprocessed Data
train_df = pd.read_csv('data/train_df.csv')
val_df = pd.read_csv('data/val_df.csv')
test_df = pd.read_csv('data/test_df.csv')

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
        torch.save(model.state_dict(), 'models/ncf_model_best.pt')
        print("Validation loss improved, saving model.")
    else:
        epochs_without_improvement += 1
        print(f"No improvement for {epochs_without_improvement} epoch(s).")

    # Stop training if no improvement for 'patience' epochs
    if epochs_without_improvement >= patience:
        print(f"Early stopping triggered after {epoch+1} epochs.")
        break

# Load best model weights after training
model.load_state_dict(torch.load('models/ncf_model_best.pt'))
