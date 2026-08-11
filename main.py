import pandas as pd
import numpy as np
from surprise.accuracy import rmse, mae
import matplotlib.pyplot as plt

from src.utils import load_and_preprocess_data
from src.models import build_svd_model, build_vae_model
from src.metrics import get_ndcg_at_k, masked_mse

"""
Öneri Sistemlerinde Derecelendirme Tahmini: 
Varyasyonel Otokodlayıcı (VAE) ile Matris Faktörizasyonunun (MF - SVD) Karşılaştırmalı Performans Analizi

Yazar: Muhammed Ali Ezgin
Danışman: Dr. Öğr. Üyesi Alper Yargıç
Bilecik Şeyh Edebali Üniversitesi - Bilgisayar Mühendisliği Bölümü
BM400 Bitirme Çalışması
"""

if __name__ == '__main__':
    # -------------------------------------------------------------------------
    # ADIM 1: Veri Ön İşleme ve Bellek Optimizasyonu
    # -------------------------------------------------------------------------
    data_dict = load_and_preprocess_data(
        file_path='ratings.dat',
        min_user_ratings=10,
        min_item_ratings=5,
        test_size=0.2,
        random_state=42
    )

    df = data_dict['df']
    trainset = data_dict['trainset']
    testset = data_dict['testset']
    user_to_index = data_dict['user_to_index']
    item_to_index = data_dict['item_to_index']
    num_users = data_dict['num_users']
    num_items = data_dict['num_items']
    full_sparse_matrix = data_dict['full_sparse_matrix']

    # -------------------------------------------------------------------------
    # ADIM 2: Matris Faktörizasyonu (MF - SVD) Eğitimi ve Değerlendirmesi
    # -------------------------------------------------------------------------
    print("\n--- ADIM II: Matris Faktörizasyonu (SVD) Kurulumu ---")
    svd_model = build_svd_model(n_factors=300, reg_all=0.05, random_state=42)

    svd_model.fit(trainset)
    predictions_mf = svd_model.test(testset)
    rmse_mf = rmse(predictions_mf, verbose=False)
    mae_mf = mae(predictions_mf, verbose=False)
    ndcg_mf = get_ndcg_at_k(predictions_mf, k=5, threshold=4.0)

    print("\n--- MF (SVD) Sonuçları ---")
    print(f"MF Model RMSE   : {rmse_mf:.4f}")
    print(f"MF Model MAE    : {mae_mf:.4f}")
    print(f"MF Model NDCG@5 : {ndcg_mf:.4f}")
    results = {'MF_RMSE': rmse_mf, 'MF_MAE': mae_mf, 'MF_NDCG': ndcg_mf}

    # -------------------------------------------------------------------------
    # ADIM 3: Varyasyonel Otokodlayıcı (VAE) Eğitimi ve Değerlendirmesi
    # -------------------------------------------------------------------------
    print("\n--- ADIM III: Varyasyonel Otokodlayıcı (VAE) Kurulumu ---")

    R_dense = full_sparse_matrix.toarray().astype(np.float32)

    test_size_vae = int(num_users * 0.2)
    train_indices = np.random.choice(num_users, num_users - test_size_vae, replace=False)
    R_train_dense = R_dense[train_indices, :]
    R_train_normalized = R_train_dense / 5.0
    R_test_dense = R_dense[np.setdiff1d(np.arange(num_users), train_indices), :]

    original_dim = num_items   # Giriş katmanı öğe sayısı
    latent_dim = 50            # Gizli faktör sayısı (z)
    intermediate_dim = 200     # Ara katman nöron sayısı
    dropout_rate = 0.4         # Dropout oranı

    vae = build_vae_model(
        original_dim=original_dim,
        latent_dim=latent_dim,
        intermediate_dim=intermediate_dim,
        dropout_rate=dropout_rate,
        beta=0.01,
        learning_rate=0.0005
    )

    new_epochs = 200
    print(f"VAE Modeli Eğitiliyor... (Epoch Sayısı: {new_epochs})")

    history = vae.fit(R_train_normalized, R_train_normalized, epochs=new_epochs, batch_size=128, shuffle=True, verbose=1)

    print("VAE Modeli Eğitimi Tamamlandı.")

    # Epoch - Loss Grafiği
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Eğitim Kaybı (Loss)')
    plt.title('VAE Model Eğitim Süreci')
    plt.xlabel('Epoch')
    plt.ylabel('Masked MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # VAE Tahminleri
    R_predicted_vae = vae.predict(R_test_dense / 5.0)

    observed_mask = (R_test_dense > 0)
    diff = (R_test_dense - R_predicted_vae * 5.0) * observed_mask
    rmse_vae = np.sqrt(np.sum(diff**2) / np.sum(observed_mask))
    mae_vae = np.sum(np.abs(diff)) / np.sum(observed_mask)

    ndcg_vae = get_ndcg_at_k((R_test_dense, R_predicted_vae), k=5, threshold=4.0, is_vae=True)

    print("\n--- VAE Sonuçları ---")
    print(f"VAE Model RMSE   : {rmse_vae:.4f}")
    print(f"VAE Model MAE    : {mae_vae:.4f}")
    print(f"VAE Model NDCG@5 : {ndcg_vae:.4f}")

    results['VAE_RMSE'] = rmse_vae
    results['VAE_MAE'] = mae_vae
    results['VAE_NDCG'] = ndcg_vae 

    # -------------------------------------------------------------------------
    # ADIM 4: Karşılaştırmalı Performans Analizi
    # -------------------------------------------------------------------------
    print("\n--- ADIM IV: Karşılaştırmalı Performans Analizi ---")

    comparison_data = {
        'MF_RMSE': results['MF_RMSE'],
        'MF_MAE': results['MF_MAE'],
        'MF_NDCG@5': results['MF_NDCG'],
        'VAE_RMSE': results['VAE_RMSE'],
        'VAE_MAE': results['VAE_MAE'],
        'VAE_NDCG@5': results['VAE_NDCG'],
    }

    comparison_df = pd.DataFrame([comparison_data]).T
    comparison_df.columns = ['Değer']

    print("\nMODELLER ARASI KARŞILAŞTIRMA (RMSE, MAE ve NDCG@5)")
    print(comparison_df)

    if results['MF_RMSE'] < results['VAE_RMSE']:
        print(f"\nSonuç: MF tahmin doğruluğu (RMSE={results['MF_RMSE']:.4f}) açısından daha başarılıdır.")
    else:
        print(f"\nSonuç: VAE tahmin doğruluğu (RMSE={results['VAE_RMSE']:.4f}) açısından daha başarılıdır.")

    num_users_final = R_dense.shape[0]
    num_items_final = R_dense.shape[1]

    print(f"\nFiltreleme Sonrası Kullanıcı Sayısı (Satır): {num_users_final}")
    print(f"Filtreleme Sonrası Ürün/Öğe Sayısı (Sütun): {num_items_final}") 
    print("\n--- İŞLEM TAMAMLANDI ---")
