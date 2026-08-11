# Öneri Sistemlerinde Derecelendirme Tahmini: Varyasyonel Otokodlayıcı ile Matris Faktörizasyonunun Karşılaştırmalı Performans Analizi

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Scikit-Surprise](https://img.shields.io/badge/Surprise-SVD-4B8BBE?style=for-the-badge)

Bu proje, açık derecelendirme tahmini (explicit rating prediction) kapsamında, **Derin Öğrenme** tabanlı **Varyasyonel Otokodlayıcı (Variational Autoencoder - VAE)** ile **Geleneksel İşbirlikçi Filtreleme** yöntemi olan **Tekil Değer Ayrışımı tabanlı Matris Faktörizasyonunun (Matrix Factorization - SVD)** performanslarını karşılaştırmalı olarak analiz etmektedir.

---

## 🎓 Proje Bilgileri

- **Yazar:** Muhammed Ali Ezgin (No: 50674850364)
- **Danışman:** Dr. Öğr. Üyesi Alper Yargıç
- **Kurum:** T.C. Bilecik Şeyh Edebali Üniversitesi, Mühendislik Fakültesi, Bilgisayar Mühendisliği Bölümü
- **Ders:** BM400 Bitirme Çalışması
- **Rapor Belgeleri:** [`docs/Bitirme_Calismasi_Raporu_Muhammed_Ali_Ezgin.pdf`](docs/Bitirme_Calismasi_Raporu_Muhammed_Ali_Ezgin.pdf)

---

## 📌 Proje Özeti ve Amacı

E-ticaret ve dijital içerik platformlarında öneri sistemleri kullanıcı deneyimini doğrudan etkilemektedir. Bu çalışmanın temel amacı:
- Seyrek (sparse) kullanıcı-öğe etkileşim matrisleri üzerinde klasik matris faktörizasyonu (SVD) ile derin olasılıksal otokodlayıcı (VAE) modellerinin davranışlarını incelemek.
- Modelleri hem **hata metrikleri** ($RMSE$, $MAE$) hem de **sıralama/tavsiye kalitesi metrikleri** ($NDCG@5$) üzerinden bütünsel olarak kıyaslamak.

---

## 📊 Veri Seti ve Ön İşleme

Çalışmada **MovieLens 1M** veri seti kullanılmıştır. Veri setindeki aşırı seyreklik (data sparsity) sorununu çözmek amacıyla aşağıdaki filtreleme kuralları uygulanmıştır:

- **Kullanıcı Filtresi:** En az **10 oy** veren kullanıcılar tutulmuştur (`min_user_ratings = 10`).
- **Öğe Filtresi:** En az **5 oy** alan ürünler/filmler tutulmuştur (`min_item_ratings = 5`).

### Veri Seti İstatistikleri
| Özellik | Sayı / Değer |
| :--- | :--- |
| **Filtrelenmiş Kullanıcı Sayısı (Satır)** | **5,624** |
| **Filtrelenmiş Ürün/Öğe Sayısı (Sütun)** | **3,413** |
| **Puan Skalası** | 1.0 - 5.0 |
| **Eğitim / Test Bölme Oranı** | %80 Eğitim / %20 Test |

---

## 🛠️ Model Mimarileri ve Parametreleri

### 1. Geleneksel Yöntem: Matris Faktörizasyonu (Surprise SVD)
- **Kütüphane:** `scikit-surprise`
- **Gizli Faktör Sayısı ($n\_factors$):** `300`
- **Düzenlileştirme Katsayısı ($reg\_all$):** `0.05`
- **Random State:** `42`

### 2. Derin Öğrenme Yöntemi: Varyasyonel Otokodlayıcı (VAE)
- **Kütüphane:** TensorFlow / Keras (Özel `SamplingAndKL` katmanı ile)
- **Girdi Boyutu:** `3,413` (Öğe sayısı)
- **Ara Katman Boyutu (Hidden/Intermediate Dim):** `200` (ReLU aktivasyonu + Dropout `0.4`)
- **Gizli Uzay Boyutu (Latent Dim $z$):** `50` ($\mu$ ve $\log\sigma^2$)
- **Kayıp Fonksiyonu:** Maskelenmiş MSE (`masked_mse`) + $\beta \cdot \mathcal{L}_{KL}$ ($\beta = 0.01$)
- **Optimizasyon Algoritması:** Adam ($Learning\ Rate = 0.0005$)
- **Epoch / Batch Size:** `200 Epoch` / `128 Batch`

---

## 📈 Deneysel Sonuçlar ve Karşılaştırma

Aşağıdaki tablo, 5,624 kullanıcı ve 3,413 öğe içeren matris üzerinde eğitilen modellerin **RMSE**, **MAE** ve **NDCG@5** sonuçlarını göstermektedir:

| Model | RMSE ↓ *(Düşük İyi)* | MAE ↓ *(Düşük İyi)* | NDCG@5 ↑ *(Yüksek İyi)* |
| :--- | :---: | :---: | :---: |
| **Matris Faktörizasyonu (SVD)** | **0.8646** 🏆 | **0.6836** 🏆 | 0.8741 |
| **Varyasyonel Otokodlayıcı (VAE)** | 0.9207 | 0.7569 | **0.9091** 🏆 |

### 💡 Ana Bulgular
1. **Nokta Tahmin Doğruluğu (RMSE / MAE):** Klasik **SVD** modeli, sayısal puan tahmin hatalarında VAE'ye göre daha düşük hata vermiş ve daha hassas sayısal tahminler üretmiştir ($RMSE = 0.8646$).
2. **Sıralama Başarısı (NDCG@5):** **VAE** modeli, olasılıksal gizli uzay temsil gücü sayesinde kullanıcıların yüksek puan vereceği ilk 5 ürünü doğru sıralama konusunda SVD'yi geride bırakmıştır ($NDCG@5 = 0.9091$).

---

## 💻 Kurulum ve Çalıştırma

### 1. Depoyu Klonlayın
```bash
git clone https://github.com/kullanici_adi/recommender-system-vae-vs-svd.git
cd recommender-system-vae-vs-svd
```

### 2. Gerekli Kütüphaneleri Yükleyin
```bash
pip install -r requirements.txt
```

### 3. Veri Setini Hazırlayın
MovieLens veri setine ait `ratings.dat` dosyasını ana dizine koyun (Format: `userID::itemID::rating::timestamp`).

### 4. Modelleri Eğitin ve Test Edin
```bash
python main.py
```

---

## 📜 Lisans ve Atıf

Bu proje Bilecik Şeyh Edebali Üniversitesi Bilgisayar Mühendisliği Bölümü BM400 Bitirme Çalışması kapsamında geliştirilmiştir.
