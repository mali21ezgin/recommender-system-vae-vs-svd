import numpy as np
import math
from collections import defaultdict
import tensorflow as tf
from tensorflow.keras import backend as K
from keras import ops

def get_ndcg_at_k(predictions_or_matrix, k=5, threshold=4.0, is_vae=False):
    """
    Normalleştirilmiş İndirgenmiş Kümülatif Kazanç (NDCG@K) Hesaplama Fonksiyonu
    """
    if not is_vae:  # MF (SVD) tahmini için
        top_n = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions_or_matrix:
            top_n[uid].append((iid, true_r, est))
            
        dataset = top_n
        
    else:  # VAE tahmini için
        R_test_dense, R_predicted_vae = predictions_or_matrix
        num_users_test, num_items = R_test_dense.shape
        dataset = defaultdict(list)
        
        for user_index in range(num_users_test):
            observed_items = np.where(R_test_dense[user_index, :] > 0)[0]  # Kullanıcının oy kullandığı ürünler
            
            for item_index in observed_items:
                true_r = R_test_dense[user_index, item_index]
                est = R_predicted_vae[user_index, item_index]
                dataset[user_index].append((item_index, true_r, est))

    ndcgs = []
     
    for uid, user_ratings in dataset.items():
        # user_ratings: (item_id, true_rating, estimated_rating)
        user_ratings.sort(key=lambda x: x[2], reverse=True)  # Tahmin edilen puana göre azalan sırada sırala

        relevance = [1.0 if r[1] >= threshold else 0.0 for r in user_ratings[:k]]
        ideal_relevance = sorted([1.0 if r[1] >= threshold else 0.0 for r in user_ratings], reverse=True)[:k]
        
        dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance))
        idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_relevance))
        
        if idcg > 0:
            ndcgs.append(dcg / idcg) 
            
    return np.mean(ndcgs) if ndcgs else 0.0


def masked_mse(y_true, y_pred):
    """
    Kullanıcının yalnızca oy verdiği öğeler üzerinden hesaplanan Maskelenmiş MSE (Mean Squared Error) Kayıp Fonksiyonu
    """
    # 0 olmayan (oy verilmiş) öğeleri maskele
    mask = ops.cast(ops.not_equal(y_true, 0), dtype=K.floatx())
    
    # Hata karelerini hesapla ve maskele
    squared_error = ops.square(y_true - y_pred) * mask
    
    # Kullanıcı başına toplam hatayı hesapla (axis=-1: satır bazında toplam)
    reconstruction_loss_per_user = ops.sum(squared_error, axis=-1)
    
    # Batch içindeki tüm kullanıcıların ortalama hatasını döndür
    return ops.mean(reconstruction_loss_per_user)
