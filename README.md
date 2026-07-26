# Aspect-Based Sentiment Analysis pada Ulasan E-commerce

## Ringkasan
Proyek ini menerapkan Aspect-Based Sentiment Analysis (ABSA) pada 5.000 ulasan produk makanan yang dikumpulkan dari sebuah marketplace grocery daring. Alih-alih mengklasifikasikan seluruh ulasan sekadar sebagai positif atau negatif, analisis ini mengidentifikasi aspek produk spesifik apa saja (rasa, harga, layanan, pengiriman, ukuran) yang dibahas pelanggan beserta sentimen yang menyertainya, kemudian melatih classifier Support Vector Machine (SVM) untuk masing-masing aspek.

## Tujuan
- Mengidentifikasi aspek yang paling sering dipuji atau dikeluhkan dalam ulasan pelanggan.
- Membangun dan mengevaluasi classifier sentimen per aspek yang dapat digunakan bisnis untuk menyaring ulasan yang masuk secara otomatis.
- Menerjemahkan distribusi sentimen per aspek menjadi rekomendasi praktis untuk pemasaran dan peningkatan layanan.

## Data
- Dataset "Food Reviews: Text Mining and Sentiment Analysis" (Kaggle), 5.000 ulasan produk grocery.
- Kolom: ProductID, Helpfulness_Score, Rating (1-5), Review_Title, dan Text.

## Metodologi
1. Preprocessing teks: case folding, penguraian kontraksi, penghapusan karakter tidak relevan (URL, mention, angka, tanda baca), penghapusan duplikat, dan tokenisasi.
2. Ekstraksi dan pemetaan aspek: ekstraksi kata benda menggunakan spaCy, dikelompokkan ke dalam lima aspek yang telah ditentukan (taste, price, service, delivery, size).
3. Penilaian sentimen per aspek menggunakan VADER, mengklasifikasikan ulasan sebagai positif (compound score > 0,05) atau negatif (compound score < -0,05); ulasan netral dikeluarkan karena dianggap tidak menyumbang opini substansial.
4. Representasi fitur menggunakan TF-IDF, dilanjutkan dengan training SVM per aspek dan evaluasi menggunakan pembagian stratified 60:40, dipilih untuk menjaga kelas minoritas (negatif) tetap cukup terwakili di kedua set.
5. Evaluasi menggunakan macro F1-score, dipilih untuk memberi bobot yang sama pada kelas mayoritas dan minoritas mengingat ketidakseimbangan kelas dalam data.

## Temuan Utama
- Kelima aspek (delivery, price, service, size, taste) menunjukkan mayoritas ulasan positif yang kuat (lebih dari 90% positif pada setiap aspek), dengan taste dan price sebagai aspek yang paling sering dibahas.
- Classifier SVM cukup andal mendeteksi sentimen positif di seluruh aspek (F1-score 0,96-0,98), namun jauh lebih lemah dalam mendeteksi sentimen negatif (F1-score berkisar dari 0,00 pada delivery hingga 0,50 pada price dan service), mencerminkan kelangkaan contoh negatif yang tersedia untuk training.
- Macro F1-score berkisar dari 0,48 (delivery) hingga 0,74 (price dan service), menunjukkan bahwa angka akurasi keseluruhan dapat menutupi kinerja yang buruk pada kelas minoritas.

## Relevansi Bisnis
Proporsi ulasan positif yang konsisten tinggi di seluruh aspek mendukung pesan pemasaran yang dibangun di sekitar rasa dan daya saing harga, serta menyoroti kecepatan pengiriman dan responsivitas layanan sebagai tema yang saling menguatkan. Di sisi lain, lemahnya deteksi kelas negatif oleh model juga merupakan temuan yang dapat ditindaklanjuti: bisnis yang mengandalkan penyaringan ulasan otomatis tidak boleh mengasumsikan sedikitnya jumlah flag negatif berarti tidak ada keluhan, dan sebaiknya melengkapi penyaringan otomatis dengan tinjauan manual berkala, terutama untuk aspek dengan sedikit contoh negatif seperti delivery.

## Tools & Libraries
Python, spaCy, VADER (SentimentIntensityAnalyzer), scikit-learn (TF-IDF, SVM), contractions, NLTK.

## Isi Repository
- `FoodReviews.csv` — dataset ulasan yang sudah dibersihkan lengkap dengan label aspek dan sentimen.
- `ABSA_SVM.ipynb` — notebook analisis lengkap (preprocessing, ekstraksi aspek, penilaian VADER, training dan evaluasi SVM).
- `README.md` — dokumentasi proyek dalam bahasa Inggris dan Indonesia.

## Keterbatasan
Pembagian train-test 60:40, yang dipilih untuk mempertahankan contoh kelas minoritas, tetap menyisakan sangat sedikit sampel negatif untuk beberapa aspek (misalnya hanya 4 pada test set delivery), sehingga membatasi keandalan penilaian kinerja classifier untuk kelas tersebut.

---------------------------------------------------------------------------------

# Aspect-Based Sentiment Analysis on E-commerce Reviews

## Overview
This project applies Aspect-Based Sentiment Analysis (ABSA) to 5,000 food product reviews collected from an online grocery marketplace. Rather than classifying an entire review as simply positive or negative, the analysis identifies which specific product aspects (taste, price, service, delivery, size) customers discuss and the sentiment attached to each one, then trains a Support Vector Machine (SVM) classifier per aspect.

## Objectives
- Identify the aspects most frequently praised or criticized within customer reviews.
- Build and evaluate a per-aspect sentiment classifier that businesses can use to automatically triage incoming reviews.
- Translate aspect-level sentiment distribution into practical recommendations for marketing and service improvement.

## Data
- "Food Reviews: Text Mining and Sentiment Analysis" dataset (Kaggle), 5,000 grocery product reviews.
- Columns: ProductID, Helpfulness_Score, Rating (1-5), Review_Title, and Text.

## Methodology
1. Text preprocessing: case folding, contraction expansion, removal of non-relevant characters (URLs, mentions, numbers, punctuation), duplicate removal, and tokenization.
2. Aspect extraction and mapping: noun extraction via spaCy, grouped into five predefined aspects (taste, price, service, delivery, size).
3. Sentiment scoring per aspect using VADER, classifying reviews as positive (compound score > 0.05) or negative (compound score < -0.05); neutral reviews were excluded as non-substantive.
4. Feature representation using TF-IDF, followed by per-aspect SVM training and evaluation with a stratified 60:40 train-test split, chosen to keep the minority (negative) class adequately represented in both sets.
5. Evaluation using macro F1-score, selected to weight majority and minority classes equally given the class imbalance in the data.

## Key Findings
- All five aspects (delivery, price, service, size, taste) showed a strong majority of positive reviews (over 90% positive in every aspect), with taste and price the most frequently discussed.
- SVM classifiers reliably detect positive sentiment across aspects (F1-scores of 0.96-0.98) but are considerably weaker at detecting negative sentiment (F1-scores ranging from 0.00 for delivery to 0.50 for price and service), reflecting the scarcity of negative examples available for training.
- Macro F1-scores range from 0.48 (delivery) to 0.74 (price and service), highlighting that overall accuracy figures can mask poor minority-class performance.

## Business Relevance
The consistently high share of positive reviews across aspects supports marketing messaging built around taste and price competitiveness, and highlights fast delivery and responsive service as reinforcing themes. At the same time, the model's weak negative-class detection is itself an actionable finding: businesses relying on automated review triage should not assume low negative-flag counts mean an absence of complaints, and should supplement automated triage with periodic manual review, especially for aspects with few negative examples such as delivery.

## Tools & Libraries
Python, spaCy, VADER (SentimentIntensityAnalyzer), scikit-learn (TF-IDF, SVM), contractions, NLTK.

## Repository Contents
- `FoodReviews.csv` — cleaned review dataset with aspect and sentiment labels.
- `ABSA_SVM.ipynb` — full analysis notebook (preprocessing, aspect extraction, VADER scoring, SVM training and evaluation).
- `README.md` — project documentation in English and Indonesian.

## Limitations
The 60:40 train-test split, chosen to preserve minority-class examples, still leaves very few negative samples for some aspects (e.g., only 4 in the delivery test set), limiting how reliably classifier performance can be assessed for that class.
