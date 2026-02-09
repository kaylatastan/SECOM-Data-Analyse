# Yarı İletken Üretim Kusur Tespiti: Makine Öğrenmesi Yaklaşımı

**UCI SECOM Veri Seti Analizi**

---

## 1. Proje Özeti

Bu proje, yarı iletken üretim süreçlerinde kusur tespiti için makine öğrenmesi (ML) pipeline'ı geliştirmeyi amaçlamaktadır. Çalışmada UCI Machine Learning Repository'den alınan **SECOM (Semiconductor Manufacturing)** veri seti kullanılmıştır. Proje, endüstriyel üretim kalite kontrolü bağlamında sınıflandırma problemlerinin çözümüne odaklanmaktadır.

### 1.1 Amaç ve Hedefler

- Yarı iletken üretim sürecinde hatalı ürünlerin önceden tespiti
- Yüksek boyutlu sensör verisinden anlamlı örüntülerin çıkarılması
- Sınıf dengesizliği probleminin etkili yönetimi
- Farklı ML algoritmalarının performans karşılaştırması
- Endüstriyel uygulamalar için yorumlanabilir modellerin geliştirilmesi

---

## 2. Veri Seti Tanımı

### 2.1 Veri Kaynağı

| Özellik | Değer |
|---------|-------|
| **Kaynak** | UCI Machine Learning Repository |
| **Veri Seti Adı** | SECOM (Semiconductor Manufacturing) |
| **Örnek Sayısı** | 1,567 |
| **Özellik Sayısı** | 591 sensör ölçümü |
| **Hedef Değişken** | Pass (Geçti) / Fail (Hatalı) |
| **Sınıf Dağılımı** | ~93.4% Pass, ~6.6% Fail |

### 2.2 Veri Seti Özellikleri

- **Yüksek Boyutluluk**: 591 adet sensör ölçümü içermektedir
- **Eksik Değerler**: Birçok sütunda yüksek oranda eksik veri bulunmaktadır
- **Sınıf Dengesizliği**: Ciddi düzeyde dengesiz sınıf dağılımı (%93.4 vs %6.6)
- **Gerçek Endüstriyel Veri**: Üretim hattından elde edilen gerçek sensör verileri

### 2.3 Referans

```
McCann, M., & Johnston, A. (2008). SECOM Dataset. UCI Machine Learning Repository.
Dua, D. and Graff, C. (2019). UCI Machine Learning Repository. Irvine, CA: University of California, School of Information and Computer Science.
```

---

## 3. Metodoloji

### 3.1 Veri Ön İşleme

#### 3.1.1 Eksik Değer Analizi ve İşleme

Veri setinde önemli miktarda eksik değer tespit edilmiştir. Eksik değer stratejisi olarak **"Informative Missingness (Bilgilendirici Eksiklik)"** yaklaşımı benimsenmiştir:

1. **Eksik Değer Tespiti**: %5'ten fazla eksik değere sahip sütunlar belirlendi
2. **Binary Flag Oluşturma**: Her yüksek eksiklik oranına sahip sütun için `_is_missing` binary gösterge değişkeni oluşturuldu
3. **Median Doldurma**: Eksik değerler median imputation yöntemi ile dolduruldu

**Neden Bu Yaklaşım?**
- Sensör arızaları genellikle üretim problemlerinin göstergesidir
- Eksiklik örüntüsü, kusur tespitinde önemli bir sinyal taşıyabilir
- 100+ adet binary flag oluşturulmuş ve bunların bir kısmı en önemli özellikler arasında yer almıştır

#### 3.1.2 Özellik Mühendisliği

| İşlem | Açıklama |
|-------|----------|
| **Numerik olmayan sütunların çıkarılması** | 'Time' sütunu veri setinden kaldırıldı |
| **Informative Missingness Flags** | 100+ adet `_is_missing` binary flag oluşturuldu |
| **StandardScaler Normalizasyonu** | Tüm özellikler standartlaştırıldı (μ=0, σ=1) |

#### 3.1.3 Hedef Değişken Dönüşümü

Orijinal etiketler (-1 = Pass, 1 = Fail) → Dönüştürülmüş etiketler (0 = Fail, 1 = Pass)

### 3.2 Sınıf Dengesizliği Yönetimi

#### 3.2.1 SMOTE (Synthetic Minority Over-sampling Technique)

Sınıf dengesizliği problemi **SMOTE** yöntemi ile ele alınmıştır:

- **Uygulama Yöntemi**: SMOTE **yalnızca çapraz doğrulama katmanları içinde** uygulanmıştır
- **Neden CV İçinde?**: Veri sızıntısını (data leakage) önlemek için
- **Kütüphane**: `imbalanced-learn` (imblearn)

```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

#### 3.2.2 Sınıf Ağırlıkları

Ağaç tabanlı modellerde `class_weight='balanced'` parametresi kullanılmıştır.

### 3.3 Veri Bölümleme

| Parametre | Değer |
|-----------|-------|
| **Eğitim/Test Oranı** | %80 / %20 |
| **Strateji** | Stratified (Tabakalı) bölümleme |
| **Random State** | 42 (tekrarlanabilirlik için) |

---

## 4. Normallik Testleri ve İstatistiksel Analiz

### 4.1 Normallik Testlerinin Önemi

Makine öğrenmesi modellemesinde normallik testleri, verilerin dağılım özelliklerini anlamak için kritik öneme sahiptir:

- Birçok istatistiksel yöntem normal dağılım varsayımına dayanır
- Preprocessing stratejilerinin belirlenmesine yardımcı olur
- Aykırı değer tespitinde referans sağlar

### 4.2 Uygulanan İstatistiksel Testler

| Test | Amaç | Uygulama |
|------|------|----------|
| **Shapiro-Wilk** | Normallik testi | Örnek boyutu küçük olduğunda tercih edilir |
| **Kolmogorov-Smirnov** | Normallik testi | Büyük veri setleri için |
| **Skewness/Kurtosis** | Dağılım şekli | Çarpıklık ve basıklık analizi |

### 4.3 Normallik Testi Sonuçları

Sensör verilerinin büyük çoğunluğu **normal dağılıma uymamaktadır**:

- Yüksek çarpıklık (skewness) değerleri gözlemlenmiştir
- Aykırı değerler ve uzun kuyruklar tespit edilmiştir
- Bu nedenle **StandardScaler** ile normalizasyon uygulanmıştır

### 4.4 Kullanılan İstatistiksel Kütüphaneler

- `scipy.stats` - İstatistiksel testler için
- `statsmodels` - İleri düzey istatistiksel analiz için

---

## 5. Kullanılan Makine Öğrenmesi Modelleri

### 5.1 Model Özeti

| Model | Algoritma Tipi | Kullanım Durumu |
|-------|---------------|-----------------|
| **Random Forest** | Ensemble (Bagging) | ✅ Kullanıldı |
| **XGBoost** | Ensemble (Gradient Boosting) | ✅ Kullanıldı |
| **Yapay Sinir Ağı (ANN)** | Deep Learning | ✅ Kullanıldı |
| **KNN** | Instance-based | ✅ Deneysel olarak test edildi |
| **SVM** | Kernel-based | ✅ Deneysel olarak test edildi |
| **PCA** | Boyut İndirgeme | ✅ Deneysel, ancak son pipeline'da kullanılmadı |

### 5.2 Random Forest

#### Hiperparametreler

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    class_weight='balanced',
    random_state=42
)
```

#### Neden Random Forest?

- **Yorumlanabilirlik**: Feature importance sağlar
- **Robustness**: Aykırı değerlere karşı dayanıklı
- **Overfitting Kontrolü**: Ensemble yapısı sayesinde generalization iyi

### 5.3 XGBoost

#### Hiperparametreler

```python
XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=auto,  # Sınıf dengesizliği için
    random_state=42
)
```

#### Neden XGBoost?

- **Yüksek Performans**: Gradient boosting tabanlı güçlü algoritma
- **Sınıf Dengesizliği Desteği**: `scale_pos_weight` parametresi
- **Regularization**: L1/L2 regularization ile overfitting kontrolü

### 5.4 Yapay Sinir Ağı (ANN)

#### Mimari

```
Input Layer:  591 nöron (özellik sayısı)
    ↓
Hidden Layer 1: 256 nöron + ReLU + Dropout(0.3)
    ↓
Hidden Layer 2: 128 nöron + ReLU + Dropout(0.3)
    ↓
Hidden Layer 3: 64 nöron + ReLU + Dropout(0.3)
    ↓
Output Layer: 1 nöron + Sigmoid
```

#### Hiperparametreler

| Parametre | Değer |
|-----------|-------|
| **Optimizer** | Adam (lr=0.001) |
| **Loss Function** | Binary Cross-Entropy |
| **Epochs** | 50 |
| **Batch Size** | 32 |
| **Dropout Rate** | 0.3 |

#### Neden PyTorch?

- Esneklik ve özelleştirme kolaylığı
- GPU desteği
- Dinamik grafik yapısı

---

## 6. Kullanılmayan/Deneysel Teknikler

### 6.1 PCA (Principal Component Analysis)

**Durum**: Deneysel notebook'ta test edildi, ancak son pipeline'da **kullanılmadı**.

#### PCA Karşılaştırma Sonuçları

| Metrik | PCA Olmadan | PCA İle (%95 varyans) |
|--------|-------------|----------------------|
| **ROC-AUC** | ~0.80 | ~0.78 |
| **Yorumlanabilirlik** | ✅ Tam | ❌ Yok |
| **Missing Flags Kullanımı** | ✅ Evet | ❌ Anlamsızlaşıyor |

#### Neden PCA Kullanılmadı?

1. **Performans Kazancı Yok**: PCA ile AUC değerinde iyileşme gözlemlenmedi
2. **Yorumlanabilirlik Kaybı**: Özellik önemliliği yorumu kayboldu
3. **Informative Missingness Flags**: Binary göstergeler PCA sonrası anlamsızlaşıyor

### 6.2 KNN (K-Nearest Neighbors)

- Deneysel notebook'ta test edildi
- Yüksek boyutlu veriler için uygun değil ("curse of dimensionality")
- Ana pipeline'da tercih edilmedi

### 6.3 SVM (Support Vector Machine)

- Deneysel analiz için kullanıldı
- Büyük veri setlerinde eğitim süresi uzun
- Ağaç tabanlı modeller daha iyi performans gösterdi

---

## 7. Model Performans Metrikleri

### 7.1 Test Seti Sonuçları

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|-------|----------|-----------|--------|----------|---------|--------|
| **Random Forest** | 0.930 | 0.933 | 0.997 | 0.964 | 0.246 | 0.977 |
| **XGBoost** | 0.930 | 0.933 | 0.997 | 0.964 | 0.297 | 0.968 |
| **ANN** | 0.924 | 0.935 | 0.986 | 0.960 | 0.340 | 0.965 |

### 7.2 Çapraz Doğrulama (5-Fold Stratified CV) Sonuçları

| Model | CV-Accuracy | CV-F1 | CV-AUC |
|-------|-------------|-------|--------|
| **Random Forest** | 0.928 ± 0.000 | 0.963 ± 0.000 | 0.337 ± 0.000 |
| **XGBoost** | 0.932 ± 0.000 | 0.965 ± 0.000 | 0.333 ± 0.000 |

### 7.3 Metrik Yorumları

#### Accuracy (Doğruluk)

- Tüm modeller ~%93 accuracy değerine ulaştı
- **Dikkat**: Dengesiz sınıflarda yanıltıcı olabilir

#### Precision (Kesinlik)

- Hatalı tahmin edilenlerin ne kadarı gerçekten hatalı?
- ~0.93 değerleri iyi performansı gösteriyor

#### Recall (Duyarlılık)

- Gerçek hatalıların ne kadarı yakalandı?
- ~0.99+ değerler çoğu hatanın tespit edildiğini gösteriyor

#### F1-Score

- Precision ve Recall'un harmonik ortalaması
- ~0.96 değerleri dengeli performansı gösteriyor

#### ROC-AUC

- Sınıflandırma eşiğinden bağımsız performans ölçümü
- 0.25-0.34 aralığı dengesiz sınıflar için expected

#### PR-AUC (Precision-Recall AUC)

- Dengesiz sınıflar için daha anlamlı
- ~0.97 değerleri güçlü performansı gösteriyor

---

## 8. Özellik Önemlilik Analizi

### 8.1 En Önemli 20 Özellik (Random Forest)

| Sıra | Özellik | Önem Skoru | Tip |
|------|---------|------------|-----|
| 1 | 486 | 0.0180 | Sensör |
| 2 | 95 | 0.0171 | Sensör |
| 3 | 351 | 0.0169 | Sensör |
| 4 | 213 | 0.0166 | Sensör |
| 5 | 103 | 0.0162 | Sensör |
| 6 | 59 | 0.0147 | Sensör |
| 7 | 33 | 0.0138 | Sensör |
| 8 | 433 | 0.0129 | Sensör |
| 9 | 28 | 0.0121 | Sensör |
| 10 | 31 | 0.0116 | Sensör |
| 11 | 559 | 0.0108 | Sensör |
| 12 | 419 | 0.0106 | Sensör |
| 13 | 247 | 0.0094 | Sensör |
| 14 | 561 | 0.0092 | Sensör |
| 15 | 511 | 0.0089 | Sensör |
| 16 | 510 | 0.0087 | Sensör |
| 17 | 205 | 0.0084 | Sensör |
| 18 | 519 | 0.0081 | Sensör |
| 19 | 563 | 0.0070 | Sensör |
| 20 | 477 | 0.0067 | Sensör |

### 8.2 Informative Missingness Flags - En Önemli Olanlar

| Özellik | Önem Skoru |
|---------|------------|
| 345_is_missing | 0.0061 |
| 72_is_missing | 0.0053 |
| 73_is_missing | 0.0052 |
| 579_is_missing | 0.0034 |
| 580_is_missing | 0.0033 |
| 581_is_missing | 0.0021 |
| 578_is_missing | 0.0019 |

**Bulgu**: Eksiklik bayrakları en önemli 30 özellik arasında yer almaktadır. Bu, "informative missingness" stratejisinin geçerliliğini doğrulamaktadır.

---

## 9. Görselleştirmeler

### 9.1 Oluşturulan Figürler

| Dosya Adı | Açıklama |
|-----------|----------|
| `01_class_distribution.png` | Sınıf dağılımı (Pass/Fail) |
| `02_missing_values_analysis.png` | Eksik değer analizi |
| `confusion_matrices.png` | Confusion matrisleri karşılaştırması |
| `roc_pr_curves.png` | ROC ve PR eğrileri |
| `feature_importance.png` | Özellik önemliliği grafiği |
| `feature_correlation_heatmap.png` | Korelasyon ısı haritası |
| `pca_variance_analysis.png` | PCA varyans analizi |
| `pca_comparison.png` | PCA karşılaştırması |
| `cost_threshold_optimization.png` | Eşik optimizasyonu |

---

## 10. Proje Yapısı ve Teknik Altyapı

### 10.1 Dizin Yapısı

```
SECOM/
├── data/
│   └── uci-secom.csv              # Ham veri seti
├── notebooks/
│   ├── 01_final_pipeline.ipynb    # Ana analiz pipeline'ı
│   ├── 01_experiments_appendix.ipynb  # Deneysel analizler (PCA, KNN, SVM)
│   └── archive/                   # Eski notebook yedekleri
├── figures/                       # Görselleştirmeler
├── reports/
│   ├── model_comparison_final.csv # Model karşılaştırma sonuçları
│   ├── feature_importance.csv     # Özellik önemliliği
│   └── neuralboost_cost_analysis.csv
├── models/                        # Kaydedilmiş modeller
├── src/                           # Kaynak kodları
├── requirements.txt               # Python bağımlılıkları
├── README.md                      # Proje açıklaması
├── model_report.md               # Detaylı model raporu
└── cv_bullets.md                 # CV için özet maddeler
```

### 10.2 Kullanılan Teknolojiler ve Kütüphaneler

| Kategori | Kütüphane | Versiyon |
|----------|-----------|----------|
| **Temel Veri Bilimi** | NumPy | ≥1.26.0 |
| | Pandas | ≥2.0.0 |
| | SciPy | ≥1.11.0 |
| **Makine Öğrenmesi** | scikit-learn | ≥1.5.0 |
| | imbalanced-learn | ≥0.12.0 |
| | XGBoost | ≥2.0.0 |
| **Derin Öğrenme** | PyTorch | ≥2.0.0 |
| **Görselleştirme** | Matplotlib | ≥3.7.0 |
| | Seaborn | ≥0.12.0 |
| **İstatistiksel Analiz** | statsmodels | ≥0.14.0 |
| **Model Kaydetme** | joblib | ≥1.3.0 |

### 10.3 Tekrarlanabilirlik

- **Random State**: 42 (tüm operasyonlarda)
- **Cross-Validation**: 5-Fold Stratified
- **Version Control**: requirements.txt ile bağımlılık yönetimi

---

## 11. Bulgular ve Sonuçlar

### 11.1 Ana Bulgular

1. **Sınıf Dengesizliği Temel Zorluk**: %6.6 hata oranı, hassas tespit zorluğu yaratmaktadır

2. **Random Forest En İyi Denge**: Performans ve yorumlanabilirlik açısından optimal sonuç

3. **Informative Missingness Değer Kattı**: Eksiklik örüntüleri tahmin gücüne katkıda bulundu

4. **SMOTE Etkili Ancak Yetersiz**: Temel veri sınırlamaları devam ediyor

5. **PCA Gereksiz**: Performans kazancı sağlamadan yorumlanabilirliği azaltıyor

### 11.2 Sınırlılıklar

- Veri setinin düşük hata oranı (%6.6) model değerlendirmesini zorlaştırıyor
- Gerçek zamanlı tahmin için optimizasyon yapılmadı
- Maliyet-hassas (cost-sensitive) analiz için eşik optimizasyonu gerekli

### 11.3 Gelecek Çalışmalar

- [ ] Maliyet duyarlı tahmin için eşik optimizasyonu
- [ ] Özellik seçimi ile boyut indirgeme
- [ ] Ensemble model birleştirme (stacking)
- [ ] Anomali tespiti yaklaşımları

---

## 12. Proje İlerleme Durumu (TODO)

### 12.1 Tamamlanan Görevler

- [x] Veri yükleme ve keşifsel analiz (EDA)
- [x] Eksik değer analizi ve işleme
- [x] Informative missingness özellik mühendisliği
- [x] SMOTE ile sınıf dengeleme (CV içinde)
- [x] Random Forest modeli implementasyonu
- [x] XGBoost modeli implementasyonu
- [x] Yapay Sinir Ağı (ANN) implementasyonu
- [x] Model değerlendirme ve karşılaştırma
- [x] Özellik önemliliği analizi
- [x] Görselleştirmeler oluşturma
- [x] PCA deneysel analiz
- [x] KNN ve SVM deneysel analiz
- [x] Model kaydetme ve serileştirme
- [x] Dokümantasyon hazırlama

### 12.2 İsteğe Bağlı İyileştirmeler

- [ ] Hiperparametre grid search optimizasyonu
- [ ] Model ensemble (stacking/blending)
- [ ] SHAP değerleri ile model açıklanabilirliği
- [ ] Streamlit/Flask ile web arayüzü
- [ ] Docker containerization

---

## 13. Akademik Referanslar

### 13.1 Veri Seti Referansları

```bibtex
@misc{secom_dataset,
  author = {McCann, M. and Johnston, A.},
  title = {SECOM Dataset},
  year = {2008},
  howpublished = {UCI Machine Learning Repository},
  url = {https://archive.ics.uci.edu/ml/datasets/SECOM}
}

@misc{uci_repository,
  author = {Dua, D. and Graff, C.},
  title = {UCI Machine Learning Repository},
  year = {2019},
  institution = {University of California, Irvine, School of Information and Computer Sciences},
  url = {https://archive.ics.uci.edu/ml}
}
```

### 13.2 Yöntem Referansları

- **SMOTE**: Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique.
- **Random Forest**: Breiman, L. (2001). Random Forests. Machine Learning.
- **XGBoost**: Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System.

---

## 14. İletişim ve Lisans

### 14.1 Proje Lisansı

Bu proje eğitim ve araştırma amaçlı geliştirilmiştir. UCI SECOM veri seti eğitim ve araştırma kullanımı için serbestçe kullanılabilir.

### 14.2 Katkıda Bulunma

Proje hakkında sorularınız için repository üzerinden issue açabilirsiniz.

---

*Son Güncelleme: Ocak 2026*
