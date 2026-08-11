# Veri Seti (Dataset) Bilgisi

Bu projede **MovieLens 1M** veri seti kullanılmaktadır.

## 📥 Veri Setini İndirme Adımları

1. [GroupLens MovieLens Datasets](https://grouplens.org/datasets/movielens/1m/) adresine gidin.
2. `ml-1m.zip` arşiv dosyasını indirin ve zipten çıkarın.
3. İçerisinden çıkan `ratings.dat` dosyasını projenin ana dizinine yerleştirin.

## 📄 Dosya Formatı

`ratings.dat` dosyası aşağıdaki formatta yapılandırılmıştır:
```text
UserID::MovieID::Rating::Timestamp
```
- `UserID`: 1 - 6040 arası kullanıcı kimliği
- `MovieID`: 1 - 3952 arası film/öğe kimliği
- `Rating`: 1 ile 5 arasındaki derecelendirme puanı
- `Timestamp`: Oy verildiği andaki POSIX zaman damgası
