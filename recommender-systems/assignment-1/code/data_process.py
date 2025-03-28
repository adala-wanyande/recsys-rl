import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load the data into a DataFrame
columns = ['userId', 'movieId', 'rating', 'timestamp']
df = pd.read_csv('../data/ratings.dat',
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

train_df.to_csv('../data/train_df.csv', index=False)
val_df.to_csv('../data/val_df.csv', index=False)
test_df.to_csv('../data/test_df.csv', index=False)
