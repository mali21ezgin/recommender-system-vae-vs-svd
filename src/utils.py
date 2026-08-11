import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split

def load_and_preprocess_data(file_path='ratings.dat', min_user_ratings=10, min_item_ratings=5, test_size=0.2, random_state=42):
    """
    MovieLens Veri Setini Yükler, Kullanıcı ve Öğe Bazlı Filtreler, Indeksler ve Matris Hazırlar.
    """
    column_names = ['user_id', 'item_id', 'rating', 'timestamp']

    print("--- ADIM I: Veri Ön İşleme ve Bellek Optimizasyonu Başlatılıyor ---")

    df = pd.read_csv(
        file_path, 
        sep='::', 
        header=None, 
        names=column_names, 
        usecols=[0, 1, 2], 
        engine='python', 
        dtype={'user_id': 'object', 'item_id': 'object', 'rating': np.float32}
    )

    # Filtreleme: En az min_user_ratings oy kullanan kullanıcılar
    user_counts = df['user_id'].value_counts()
    valid_users = user_counts[user_counts >= min_user_ratings].index
    df = df[df['user_id'].isin(valid_users)].copy()

    # Filtreleme: En az min_item_ratings oy alan öğeler
    item_counts = df['item_id'].value_counts()
    valid_items = item_counts[item_counts >= min_item_ratings].index
    df = df[df['item_id'].isin(valid_items)].copy()

    # Surprise Kütüphanesi Formatına Dönüştürme
    reader = Reader(rating_scale=(1.0, 5.0))
    data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=test_size, random_state=random_state)

    # Indeksleme işlemleri
    user_to_index = {user: i for i, user in enumerate(df['user_id'].unique())}
    df['user_index'] = df['user_id'].map(user_to_index).astype(np.int32)
    item_to_index = {item: i for i, item in enumerate(df['item_id'].unique())}
    df['item_index'] = df['item_id'].map(item_to_index).astype(np.int32)

    num_users = df['user_index'].nunique()
    num_items = df['item_index'].nunique()
    full_sparse_matrix = csr_matrix((df['rating'], (df['user_index'], df['item_index'])), shape=(num_users, num_items))

    return {
        'df': df,
        'trainset': trainset,
        'testset': testset,
        'user_to_index': user_to_index,
        'item_to_index': item_to_index,
        'num_users': num_users,
        'num_items': num_items,
        'full_sparse_matrix': full_sparse_matrix
    }
