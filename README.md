# Laporan Proyek Machine Learning: Prediksi Biaya Asuransi Kesehatan - Putu Yoga Suartana

## Domain Proyek

**Latar Belakang**

Industri asuransi merupakan pilar penting dalam sistem ekonomi dan kesehatan modern. Salah satu tantangan fundamental yang dihadapi oleh perusahaan asuransi kesehatan adalah aktuaria, yaitu proses menghitung dan mengelola risiko finansial. Penentuan premi yang akurat adalah kunci untuk menjaga profitabilitas perusahaan sekaligus menawarkan produk yang adil dan kompetitif kepada nasabah. Pendekatan machine learning telah terbukti dapat meningkatkan akurasi model prediktif dalam praktik aktuaria dan manajemen risiko asuransi [[1](https://www.researchgate.net/profile/Munashe-Naphtali-Mupa/publication/389132064_Machine_Learning_in_Actuarial_Science_Enhancing_Predictive_Models_for_Insurance_Risk_Management/links/67b60a83645ef274a4897f9a/Machine-Learning-in-Actuarial-Science-Enhancing-Predictive-Models-for-Insurance-Risk-Management.pdf)].

Besaran premi sangat bergantung pada estimasi biaya medis (klaim) yang akan dikeluarkan oleh seorang individu di masa depan. Biaya ini dipengaruhi oleh berbagai faktor risiko, termasuk faktor demografis (usia), kondisi kesehatan (misalnya, Indeks Massa Tubuh/BMI), dan pilihan gaya hidup (misalnya, kebiasaan merokok). Oleh karena itu, kemampuan untuk memprediksi biaya ini secara akurat menggunakan pendekatan data-driven menjadi aset strategis yang sangat berharga. Hal ini sejalan dengan berbagai penelitian yang telah berhasil menerapkan machine learning pada data rekam medis elektronik untuk tugas-tugas prediksi klinis yang kompleks [[2](https://www.atsjournals.org/doi/full/10.1513/AnnalsATS.201710-787OC)].

**Mengapa dan Bagaimana Masalah Harus Diselesaikan**

Masalah ini harus diselesaikan untuk menciptakan sistem asuransi yang lebih efisien dan adil. Dengan model prediksi yang akurat, perusahaan dapat:

1. Menetapkan Harga yang Lebih Tepat (Precision Pricing): Menawarkan premi yang benar-benar mencerminkan profil risiko individu.
2. Manajemen Risiko yang Lebih Baik: Mengidentifikasi kelompok nasabah berisiko tinggi dan mengembangkan strategi mitigasi.
3. Meningkatkan Kepuasan Pelanggan: Memberikan transparansi tentang faktor-faktor yang mempengaruhi biaya.

Solusinya adalah dengan menerapkan Machine Learning, khususnya model regresi, untuk menganalisis data historis dan membangun fungsi prediktif yang dapat mengestimasi biaya medis berdasarkan atribut-atribut nasabah.

**Referensi Terkait**

[1] M. N. Mupa, S. Tafirenyika, M. R. Nyajeka, T. M. Moyo, and E. K. Zhuwankinyus, "Machine learning in actuarial science: Enhancing predictive models for insurance risk management," IRE Journals, vol. 8, no. 8, Feb. 2025.

[2] J. C. Rojas, K. A. Carey, D. P. Edelson, L. R. Venable, M. D. Howell, and M. M. Churpek, "Predicting intensive care unit readmission with machine learning using electronic health record data," Ann. Am. Thorac. Soc., vol. 15, no. 8, pp. 954-961, Aug. 2018, doi: 10.1513/AnnalsATS.201712-923OC.

## Business Understanding
### Problem Statements

1. Bagaimana cara mengestimasi tagihan biaya medis (klaim) seorang calon nasabah secara akurat berdasarkan profil demografis dan kesehatannya?
2. Faktor-faktor apa sajakah (misalnya usia, BMI, status merokok) yang memiliki pengaruh paling signifikan terhadap peningkatan biaya medis?
3. Di antara beberapa model regresi standar, model manakah yang menawarkan keseimbangan terbaik antara akurasi prediksi dan generalisasi pada data baru?

### Goals

1. Mengembangkan sebuah model regresi machine learning yang dapat memprediksi variabel `charges` (biaya) dengan tingkat kesalahan (error) serendah mungkin.
2. Mengidentifikasi dan mengkuantifikasi pengaruh dari fitur-fitur kunci seperti `age`, `bmi`, dan `smoker` terhadap biaya asuransi.
3. Mencapai performa model yang solid, dengan target metrik R-squared (R^2) di atas 0.85.

**Rubrik/Kriteria Tambahan (Opsional)**:

### Solution statements
Untuk mencapai tujuan yang telah ditetapkan, solusi yang diajukan adalah sebagai berikut:

1. Mengimplementasikan dan Membandingkan Beberapa Algoritma Regresi: Proyek ini akan membangun, melatih, dan mengevaluasi tiga model machine learning yang berbeda untuk menemukan solusi terbaik:
    * Linear Regression: Digunakan sebagai model baseline.
    * Random Forest Regressor: Model ensemble yang kuat.
    * Gradient Boosting Regressor: Model ensemble lain yang seringkali memberikan akurasi tinggi.
2. Mengukur Kinerja dengan Metrik Standar: Kinerja dari setiap model akan diukur dan dibandingkan secara objektif menggunakan metrik evaluasi regresi, yaitu Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), dan R-squared (R^2). Model dengan nilai error terendah dan R^2 tertinggi akan dipilih sebagai solusi akhir yang paling optimal.

## Data Understanding
Dataset yang digunakan dalam proyek ini adalah "Medical Cost Personal Datasets" yang bersumber dari platform Kaggle.

* Tautan Sumber Data: [Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance)
* Informasi Data: Dataset ini terdiri dari 1338 baris dan 7 kolom. Data ini dalam kondisi sangat baik dan tidak memiliki nilai yang hilang (missing values).  

### Variabel-variabel pada Medical Cost Personal dataset adalah sebagai berikut:
* `age`: (Numerik) Usia nasabah.
* `sex`: (Kategorikal) Jenis kelamin.
* `bmi`: (Numerik) Indeks Massa Tubuh.
* `children`: (Numerik) Jumlah anak/tanggungan.
* `smoker`: (Kategorikal) Status merokok.
* `region`: (Kategorikal) Wilayah tempat tinggal.
* `charges`: (Numerik, Target) Total biaya medis.

**Exploratory Data Analysis (EDA)**

* Distribusi Biaya: Histogram dari `charges` menunjukkan distribusi yang miring ke kanan, menandakan adanya beberapa kasus dengan biaya yang sangat tinggi.
* Pengaruh Status Merokok: Visualisasi box plot dengan jelas menunjukkan bahwa `smoker` adalah faktor paling berpengaruh.

## Data Preparation
1. One-Hot Encoding:
    * Alasan: Algoritma machine learning memerlukan input numerik. Teknik ini mengubah fitur kategorikal menjadi format biner yang dapat diproses model.
2. Train-Test Split:
    * Alasan: Memisahkan data untuk melatih dan menguji model secara terpisah, guna memastikan evaluasi yang objektif dan menghindari overfitting.
3. Standardisasi Fitur:
    * Alasan: Fitur numerik memiliki skala yang berbeda. Standardisasi menyamakan skala agar setiap fitur memiliki kontribusi yang setara, yang penting untuk beberapa model seperti Regresi Linear.

## Modeling
Tiga model machine learning dilatih untuk memprediksi `charges`.

1. Linear Regression:
    * Parameter: Menggunakan parameter default dari `scikit-learn`.
    * Kelebihan: Cepat, sederhana, dan hasilnya sangat mudah diinterpretasikan.
    * Kekurangan: Tidak mampu menangkap pola non-linear yang kompleks.
2. Random Forest Regressor:
    * Parameter: `n_estimators=100`, `random_state=42`. `n_estimators` menentukan jumlah pohon dalam "hutan", sementara random_state memastikan hasil yang dapat direproduksi.
    * Kelebihan: Sangat baik dalam menangani hubungan non-linear, robust terhadap outlier, dan cenderung tidak overfitting.
    * Kekurangan: Lebih sulit diinterpretasikan (bersifat black box) dan membutuhkan lebih banyak sumber daya komputasi.
3. Gradient Boosting Regressor:
    * Parameter: `n_estimators=100`, `random_state=42`. Parameter yang sama dengan Random Forest digunakan untuk perbandingan yang adil.
    * Kelebihan: Umumnya memberikan tingkat akurasi prediksi yang sangat tinggi melalui pembelajaran sekuensial yang memperbaiki kesalahan.
    * Kekurangan: Sensitif terhadap hyperparameter dan bisa overfitting jika tidak diatur dengan baik.

Pemilihan Model Terbaik: Model terbaik dipilih berdasarkan hasil evaluasi kuantitatif. Model yang menghasilkan nilai R-squared tertinggi dan RMSE terendah akan dianggap sebagai solusi terbaik.

## Evaluation
**Metrik Evaluasi**

1. Mean Absolute Error (MAE):

    * Formula: $$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
    * Penjelasan: MAE adalah rata-rata dari nilai absolut selisih antara nilai prediksi dan nilai aktual. Metrik ini memberikan gambaran tentang besarnya kesalahan prediksi dalam satuan asli (Dolar).
  
2. Root Mean Squared Error (RMSE):

    * Formula: $$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$
    * Penjelasan: RMSE adalah akar kuadrat dari rata-rata kesalahan kuadrat. Memberikan bobot lebih pada kesalahan prediksi yang besar.

3. R-squared (R^2):

    * Formula: $$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$
    * Penjelasan: R-squared (Koefisien Determinasi) adalah metrik statistik yang mengukur proporsi varians pada variabel target yang dapat dijelaskan oleh model. Nilai mendekati 1 menunjukkan model yang lebih baik.

**Hasil Evaluasi**

Tabel berikut merangkum kinerja dari ketiga model pada data uji. Nilai ini diambil langsung dari output notebook revisi untuk memastikan konsistensi penuh.

|Model             |MAE         |MSE             |RMSE        |R-squared |
|------------------|------------|----------------|------------|----------|
|Linear Regression |4181.1945   |33596915.8514   |5796.2847   |0.7836    |
|Random Forest	    |2543.9758   |20864569.5134   |4567.7751   |0.8656    |
|Gradient Boosting |2443.4833   |18745176.4759   |4329.5700   |0.8793    |

**Analisis Hasil dan Dampak pada Business Goals**

Hasil evaluasi ini secara langsung menjawab Problem Statements dan Goals yang telah ditetapkan:

* Menjawab Problem Statement 1 & 3: Model Gradient Boosting terbukti menjadi model terbaik untuk mengestimasi biaya medis secara akurat, dengan RMSE sebesar $4337.90. Ini menjawab pertanyaan tentang cara mengestimasi biaya dan model mana yang terbaik.
* Menjawab Problem Statement 2: Tahap EDA mengonfirmasi bahwa status merokok adalah faktor paling berpengaruh, menjawab pertanyaan tentang faktor-faktor signifikan.
* Mencapai Goals 1 & 3: Tujuan untuk menghasilkan model dengan error serendah mungkin tercapai dengan memilih Gradient Boosting. Target R-squared di atas 0.85 juga terlampaui, dengan pencapaian 0.8756.
* Dampak Solution Statement: Pendekatan membandingkan tiga model terbukti sangat efektif. Ini menunjukkan bahwa investasi pada model yang lebih kompleks (Gradient Boosting) memberikan peningkatan performa yang signifikan (peningkatan R-squared sebesar 9.2% dari baseline Linear Regression), yang dapat diterjemahkan menjadi estimasi risiko yang jauh lebih akurat bagi bisnis.
