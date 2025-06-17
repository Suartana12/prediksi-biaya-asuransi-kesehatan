# Laporan Proyek Machine Learning - Putu Yoga Suartana
## Domain Proyek
**Latar Belakang**
Industri asuransi merupakan pilar penting dalam sistem ekonomi dan kesehatan modern. Salah satu tantangan utama yang dihadapi oleh perusahaan asuransi kesehatan adalah aktuaria, yaitu proses menghitung dan mengelola risiko finansial. Penentuan premi yang akurat adalah kunci untuk menjaga profitabilitas perusahaan sekaligus menawarkan produk yang adil dan kompetitif kepada nasabah. Jika premi terlalu rendah, perusahaan berisiko mengalami kerugian. Sebaliknya, jika premi terlalu tinggi, produk menjadi tidak menarik bagi calon nasabah [[1] (https://www.researchgate.net/profile/Munashe-Naphtali-Mupa/publication/389132064_Machine_Learning_in_Actuarial_Science_Enhancing_Predictive_Models_for_Insurance_Risk_Management/links/67b60a83645ef274a4897f9a/Machine-Learning-in-Actuarial-Science-Enhancing-Predictive-Models-for-Insurance-Risk-Management.pdf)].
Besaran premi sangat bergantung pada estimasi biaya medis (klaim) yang akan dikeluarkan oleh seorang individu di masa depan. Biaya ini dipengaruhi oleh berbagai faktor risiko, termasuk faktor demografis (usia), kondisi kesehatan (misalnya, Indeks Massa Tubuh/BMI), dan pilihan gaya hidup (misalnya, kebiasaan merokok). Oleh karena itu, kemampuan untuk memprediksi biaya ini secara akurat menggunakan pendekatan data-driven menjadi aset strategis yang sangat berharga[[2] (https://www.atsjournals.org/doi/full/10.1513/AnnalsATS.201710-787OC)].

**Mengapa dan Bagaimana Masalah Harus Diselesaikan**
Masalah ini harus diselesaikan untuk menciptakan sistem asuransi yang lebih efisien dan adil. Dengan model prediksi yang akurat, perusahaan dapat:
1.  **Menetapkan Harga yang Lebih Tepat (Precision Pricing):** Menawarkan premi yang benar-benar mencerminkan profil risiko individu.
2.  **Manajemen Risiko yang Lebih Baik:** Mengidentifikasi kelompok nasabah berisiko tinggi dan mengembangkan strategi mitigasi.
3.  **Meningkatkan Kepuasan Pelanggan:** Memberikan transparansi tentang faktor-faktor yang mempengaruhi biaya.
Solusinya adalah dengan menerapkan Machine Learning, khususnya model regresi, untuk menganalisis data historis dan membangun fungsi prediktif yang dapat mengestimasi biaya medis berdasarkan atribut-atribut nasabah.

**Referensi Terkait**

## Business Understanding
**Problem Statements (Pernyataan Masalah)**
1.  Bagaimana cara mengestimasi tagihan biaya medis (klaim) seorang calon nasabah secara akurat berdasarkan profil demografis dan kesehatannya?
2,  Faktor-faktor apa sajakah (misalnya usia, BMI, status merokok) yang memiliki pengaruh paling signifikan terhadap peningkatan biaya medis?
3.  Di antara beberapa model regresi standar, model manakah yang menawarkan keseimbangan terbaik antara akurasi prediksi dan generalisasi pada data baru?
**Goals (Tujuan)**
1.  Mengembangkan sebuah model regresi machine learning yang dapat memprediksi variabel charges (biaya) dengan tingkat kesalahan (error) serendah mungkin.
2.  Mengidentifikasi dan mengkuantifikasi pengaruh dari fitur-fitur kunci seperti age, bmi, dan smoker terhadap biaya asuransi.
3.  Mencapai performa model yang solid, dengan target metrik R-squared (R^2) di atas 0.85, yang menandakan model mampu menjelaskan lebih dari 85% variabilitas data.
**Solution Statement (Pernyataan Solusi)**
Untuk mencapai tujuan yang telah ditetapkan, solusi yang diajukan adalah sebagai berikut:
1.  **Mengimplementasikan dan Membandingkan Beberapa Algoritma Regresi:** Proyek ini akan membangun, melatih, dan mengevaluasi tiga model machine learning yang berbeda untuk menemukan solusi terbaik:
    *  **Linear Regression:** Digunakan sebagai model baseline untuk mengukur performa dasar.
    *  **Random Forest Regressor:** Model ensemble yang kuat untuk menangkap hubungan non-linear.
    *  **Gradient Boosting Regressor:** Model ensemble lain yang seringkali memberikan akurasi tinggi melalui proses pembelajaran sekuensial.
2.  **Mengukur Kinerja dengan Metrik Standar:** Kinerja dari setiap model akan diukur dan dibandingkan secara objektif menggunakan metrik evaluasi regresi, yaitu Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), dan R-squared (R^2). Model dengan nilai error terendah dan R^2 tertinggi akan dipilih sebagai solusi akhir yang paling optimal.

## Data Understanding
Dataset yang digunakan dalam proyek ini adalah "Medical Cost Personal Datasets" yang bersumber dari platform Kaggle.
  *  Tautan Sumber Data: https://www.kaggle.com/datasets/mirichoi0218/insurance
  *  Informasi Data: Dataset ini terdiri dari 1338 baris dan 7 kolom. Berdasarkan analisis awal (df.info()), data ini dalam kondisi sangat baik dan tidak memiliki nilai yang hilang (missing values).
**Variabel-Variabel pada Data**
  *  age: (Numerik) Usia nasabah.
  *  sex: (Kategorikal) Jenis kelamin ('female' atau 'male').
  *  bmi: (Numerik) Indeks Massa Tubuh.
  *  children: (Numerik) Jumlah anak/tanggungan.
  *  smoker: (Kategorikal) Status merokok ('yes' atau 'no').
  *  region: (Kategorikal) Wilayah tempat tinggal nasabah.
  *  charges: (Numerik, Target) Total biaya medis yang ditagihkan.
**Exploratory Data Analysis (EDA)**
Beberapa temuan kunci dari tahap EDA:
Distribusi Biaya: Histogram dari charges menunjukkan distribusi yang sangat miring ke kanan (right-skewed), menandakan bahwa sebagian besar nasabah memiliki biaya rendah, namun ada beberapa kasus dengan biaya yang sangat tinggi.
Pengaruh Status Merokok: Visualisasi box plot dengan jelas menunjukkan bahwa smoker adalah faktor paling berpengaruh. Median biaya untuk perokok jauh lebih tinggi daripada non-perokok.
Korelasi Fitur: Matriks korelasi menunjukkan bahwa age dan bmi memiliki korelasi positif dengan charges, meskipun tidak sekuat pengaruh dari status merokok.
