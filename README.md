# Pendekatan Explainable Machine Learning Menggunakan Catboost dan SHAP untuk Klasifikasi Spesies Penguin Berdasarkan Morfologi (Ciri Fisik)

Sumber Data: [Kaggle - Dataset Penguin Palmer](https://www.kaggle.com/datasets/parulpandey/palmer-archipelago-antarctica-penguin-data)

Sumber Kode: [Python - Project](https://colab.research.google.com/drive/1gbB-GAFfDWIfZFhjak70xbbphgER1Hpl?usp=sharing)

## Abstraksi
Data penguin di Kepulauan Palmer mencerminkan variasi biologis antarspesies yang kompleks dan
sering digunakan dalam penelitian ilmiah. Namun, penelitian terdahulu umumnya hanya berfokus pada
peningkatan akurasi model tanpa menjelaskan faktor-faktor yang memengaruhi hasil klasifikasi. Penelitian ini menerapkan algoritma Catboost Classifier dengan konfigurasi parameter default dan hasil hyperparameter tuning. Tahapan penelitian mencakup eksplorasi, preprocessing data, pemodelan,
evaluasi kinerja, serta interpretasi model menggunakan pendekatan Explainable Machine Learning berbasis SHAP. Model hasil tuning mencapai akurasi 99%. Analisis SHAP mengungkap bahwa fitur morfologi berkontribusi berbeda dalam membedakan spesies. Penelitian ini menghasilkan model yang akurat sekaligus interpretatif, sehingga bermanfaat untuk identifikasi spesies serta mendukung kajian keanekaragaman hayati dan konservasi.

## 1. Pendahuluan
Perkembangan machine learning telah memberikan kontribusi besar dalam berbagai bidang ilmiah, termasuk dalam upaya memahami keanekaragaman hayati, salah satunya melalui klasifikasi spesies
hewan berdasarkan karakteristik morfologi. Data penguin di Kepulauan Palmer merupakan salah satu dataset yang sering digunakan dalam penelitian ilmiah karena mencerminkan variasi biologis yang kompleks antarspesies. Namun, sebagian besar penelitian terdahulu hanya berfokus pada peningkatan akurasi model tanpa menjelaskan faktor-faktor yang memengaruhi hasil klasifikasi. Oleh karena itu, penelitian ini hadir untuk mengisi celah tersebut dengan menerapkan Catboost Classifier dalam dua konfigurasi, yaitu parameter default dan hasil hyperparameter tuning, yang diinterpretasikan
menggunakan pendekatan Explainable Artificial Intelligence berbasis SHAP guna menghasilkan model yang tidak hanya akurat, tetapi juga interpretatif.

## 2. Metode Penelitian
Pada penelitian ini memiliki serangkaian alur penelitian sistematis yang akan menggambarkan alur
secara menyeluruh mulai dari awal penelitian sampai akhir. Proses penelitian ini akan ditampilkan secara visual melalui diagram alur yang ditampilkan pada Gambar 2.1

<img src="https://github.com/cornelia128/Pendekatan-Explainable-Machine-Learning-untuk-Klasifikasi-Spesies-Penguin-Berdasarkan-Morfologi/blob/main/images/Alur%20Penelitian%20(Horizontal).png?raw=true" width=500 height=300>

Penelitian ini menggunakan dataset penguin yang diunduh dari platform Kaggle sebagai sumber data sekunder. Dataset memiliki 344 sampel dengan fitur morfologi berupa Culmen Length (mm), Culmen Depth (mm), Flipper Length (mm), dan Body Mass (g). Setelah data terkumpul, dilakukan Exploratory Data Analysis (EDA) untuk memeriksa distribusi data, ketidakseimbangan kelas, dan missing
value, yang hasilnya dijadikan dasar tahap preprocessing. Preprocessing mencakup penanganan
missing value menggunakan nilai modus untuk fitur kategorikal dan median untuk fitur numerik, encoding data kategorikal, normalisasi fitur numerik, serta penyeimbangan kelas menggunakan metode ADASYN. Dataset kemudian dibagi menjadi data latih dan data uji menggunakan metode Holdout
Validation dengan skenario rasio 60:40 dan 80:20, serta dievaluasi kestabilannya melalui k-fold Cross Validation dengan nilai k sebesar 5 dan 10. Tahap pemodelan diawali dengan membangun model
Catboost Classifier menggunakan parameter default sebagai baseline, kemudian dilanjutkan dengan
hyperparameter tuning menggunakan Bayesian Optimization berbantuan pustaka Optuna guna memperoleh konfigurasi optimal yang menghasilkan kinerja lebih baik dan stabil. Kedua konfigurasi
model tersebut dievaluasi menggunakan metri accuracy, precision, recall, dan F1-score yang
diturunkan dari confusion matrix, sehingga dapat diketahui konfigurasi mana yang memberikan
performa terbaik dalam mengklasifikasikan spesies penguin. Selanjutnya, tahap interpretasi model
dilakukan menggunakan pendekatan Explainable Machine Learning berbasis SHAP untuk mengidenti
fikasi kontribusi masing-masing fitur morfologi terhadap Keputusan prediksi model CatBoostClassifi
er, sehingga menghasilkan model yang tidak hanya akurat, tetapi juga transparan dan mudah dipahami.

## 3. Hasil dan Pembahasan
## 3.1 Exploratory Data Analysis
Pada tahap ini dilakukan analisis terhadap distribusikelas target (Species) pada dataset untuk memahami karakteristik data sebelum proses pemodelan. 

<img src="https://github.com/cornelia128/Pendekatan-Explainable-Machine-Learning-untuk-Klasifikasi-Spesies-Penguin-Berdasarkan-Morfologi/blob/main/images/Distribusi%20Target%20(Spesies).png?raw=true" width=500 height=300>

Pada Dataset, spesies Adelie Penguin memiliki 152 sampel (44,2%), Gentoo Penguin sebanyak 124 sampel (36,0%), dan Chinstrap Penguin sebanyak 68 sampel (19,8%).

## 3.2 Preprocessing
Pada tahapan ini mencakup empat proses utama, yaitu penanganan missing value, encoding, normalisasi, dan balancing data. Nilai yang hilang pada fitur kategorikal diisi menggunakan nilai modus dan kemudian dikonversi ke bentuk numerik melalui Label Encoding, sedangkan fitur numerik diisi menggunakan nilai median karena lebih robust terhadap outlier. Variabel target Species turut dienkode menggunakan Label Encoding, di mana Adelie Penguin dikodekan sebagai 0, Chinstrap Penguin sebagai 1, dan Gentoo Penguin sebagai 2. Selanjutnya, seluruh fitur numerik dinormalisasi menggunakan standard scaling sehingga menghasilkan distribusi dengan mean mendekati 0 dan standar deviasi mendekati 1 pada kedua dataset. Terakhir, ketidakseimbangan kelas ditangani menggunakan metode ADASYN, yang menghasilkan peningkatan total sampel yaitu 344 menjadi 456 data dengan distribusi kelas yang lebih proporsional, sehingga seluruh data siap digunakan pada tahap pemodelan.

## 3.3 Pembagian Dataset (Modeling)
Pada tahap pemodelan dengan parameter default menggunakan metode Holdout Validation, Dataset
dibagi dengan rasio 60:40 yang ditunjukkan pada Gambar 3.1 dan Gambar 3.2.

<img src="https://github.com/cornelia128/Pendekatan-Explainable-Machine-Learning-untuk-Klasifikasi-Spesies-Penguin-Berdasarkan-Morfologi/blob/main/images/6040.png?raw=true" width=500 height=300>

<img src="https://github.com/cornelia128/Pendekatan-Explainable-Machine-Learning-untuk-Klasifikasi-Spesies-Penguin-Berdasarkan-Morfologi/blob/main/images/CM%20Default%206040.png?raw=true" width=500 height=300>

Hasil evaluasi menunjukkan performa yang mencapai akurasi 99% dengan hampir seluruh sampel
uji berhasil diklasifikasikan dengan benar, yaitu 60 data Adelie, 61 data Chinstrap, dan 61 data Gentoo, dengan hanya satu kesalahan klasifikasi, yang mengindikasikan bahwa fitur morfologi antar kelas terpisah secara jelas.

Tahap selanjutnya dengan metode K-Fold Cross Validation menggunakan 5-Fold dilakukan untuk memperoleh estimasi performa yang lebih robust dan mengurangi potensi bias akibat pembagian data tunggal yang ditunjukkan pada Gambar 3.5 dan Gambar 3.6.

<img src="https://github.com/cornelia128/Pendekatan-Explainable-Machine-Learning-untuk-Klasifikasi-Spesies-Penguin-Berdasarkan-Morfologi/blob/main/images/5%20FOLD.png?raw=true" width=500 height=300>

<img src="https://github.com/cornelia128/Pendekatan-Explainable-Machine-Learning-untuk-Klasifikasi-Spesies-Penguin-Berdasarkan-Morfologi/blob/main/images/CM%20KFold%205.png?raw=true" width=500 height=300>

Model mencapai akurasi tertinggi sebesar 99,34% dengan standar deviasi 0,005, di mana hampir seluru
h sampel diklasifikasikan dengan benar, yaitu 150 data Adelie, 151 data Chinstrap, dan 152 data Gentoo, dengan total misklasifikasi yang sangat kecil, menunjukkan keseimbangan optimal antara akurasi dan kestabilan model.

## 3.3.1 Hyperparameter Tuning
Proses hyperparameter tuning dilakukan menggunakan Bayesian Optimization berbantuan framework
Optuna dengan 10 trial yang dievaluasi melalui skema 5-Fold Cross Validation dengan metrik akurasi.
Hasil tuning ditunjukkan pada Gambar 3.9 dan Gambar 3.10.

<img src="https://github.com/cornelia128/Pendekatan-Explainable-Machine-Learning-untuk-Klasifikasi-Spesies-Penguin-Berdasarkan-Morfologi/blob/main/images/Tuned.png?raw=true" width=500 height=300>

<img src="https://github.com/cornelia128/Pendekatan-Explainable-Machine-Learning-untuk-Klasifikasi-Spesies-Penguin-Berdasarkan-Morfologi/blob/main/images/CM%20Tuned.png?raw=true" width=500 height=300>

Menghasilkan performa yang sangat optimal dengan akurasi 99%, di mana hampir seluruh data
diklasifikasikan dengan benar dengan hanya 1 misklasifikasi dari 183 data, serta nilai precision,
recall, dan F1-Score yang konsisten mendekati sempurna di seluruh kelas.

## 3.5 Evaluasi Model
Berdasarkan hasil evaluasi pada dataset Penguin Palmer, model menunjukkan tingkat konsistensi yang baik model default dengan rasio 60:40 maupun model hasil tuning menghasilkan akurasi yang identik sebesar 99% dengan nilai precision, recall, dan F1-Score yang tinggi dan stabil di seluruh kelas, mengindikasikan bahwa model sangat andal dan mampu melakukan generalisasi dengan baik tanpa indikasi overfitting maupun underfitting.

## 3.6 Interpretasi Model
Analisis fitur terpenting berdasarkan morfologi (Ciri Fisik) global menunjukkan pola kontribusi fitur. Pada morfologi global yang ditujukkan pada Gambar 3.13.

<img src="https://github.com/cornelia128/Pendekatan-Explainable-Machine-Learning-untuk-Klasifikasi-Spesies-Penguin-Berdasarkan-Morfologi/blob/main/images/Morfologi%20Global.png?raw=true" width=500 height=300>

Memperlihatkan bahwa Culmen Length (mm) menjadi fitur paling dominan dengan kontribusi rata-
rata tertinggi mendekati 2,2, diikuti Flipper Length (mm) sekitar 1,0, Culmen Depth (mm)
sekitar 0,9, dan Body Mass (g) sebagai fitur dengan kontribusi terendah sekitar 0,35, yang mengindikasikan bahwa ukuran paruh merupakan ciri morfologi paling membedakan antar spesies.

## 3.7 Deployment
Pada tahap ini, deployment menggunakan Flask dilakukan untuk mengimplementasikan model klasifikasi spesies penguin ke dalam bentuk aplikasi berbasis web sehingga proses prediksi dapat dilakukan secara langsung melalui antarmuka yang interaktif dan mudah digunakan. Pada halaman utama yang 
ditunjukkan pada Gambar 4.43, tersedia formulir untuk memasukkan nilai parameter morfologi yang terdiri dari Culmen Length, Culmen Depth, Flipper Length, dan Body Mass. Setelah seluruh data terisi, tombol “Tentukan Spesies”, ditekan dan sistem secara otomatis memproses data tersebut untuk menghasilkan prediksi spesies.

<img src="https://github.com/cornelia128/Pendekatan-Explainable-Machine-Learning-untuk-Klasifikasi-Spesies-Penguin-Berdasarkan-Morfologi/blob/main/images/Dashboard.png?raw=true" width=500 height=300>

Hasil prediksi kemudian ditampilkan pada halaman yang sama dalam bentuk nama spesies, disertai interpretasi morfologi yang menjelaskan karakteristik fisik utama spesies tersebut serta gambar pendukung untuk memperjelas hasil klasifikasi, seperti yang ditunjukkan pada Gambar 4.44, Gambar 4.45, dan Gambar 4.46. Apabila ingin melakukan prediksi ulang, tombol “Reset” dapat digunakan 
untuk mengosongkan formulir dan memasukkan data baru. 

<img src="https://github.com/cornelia128/Pendekatan-Explainable-Machine-Learning-untuk-Klasifikasi-Spesies-Penguin-Berdasarkan-Morfologi/blob/main/images/Adelie.png?raw=true" width=300 height=600>

<img src="https://github.com/cornelia128/Pendekatan-Explainable-Machine-Learning-untuk-Klasifikasi-Spesies-Penguin-Berdasarkan-Morfologi/blob/main/images/Chinstrap.png?raw=true" width=300 height=600>

<img src="https://github.com/cornelia128/Pendekatan-Explainable-Machine-Learning-untuk-Klasifikasi-Spesies-Penguin-Berdasarkan-Morfologi/blob/main/images/Gentoo.png?raw=true" width=300 height=600>

Secara keseluruhan, implementasi model dalam bentuk aplikasi web menunjukkan bahwa sistem dapat berfungsi dengan baik, responsif terhadap input pengguna, serta memudahkan proses klasifikasi spesies penguin secara praktis dan efisien. 

## 4. Kesimpulan
Berdasarkan hasil penelitian yang telah dilakukan, dapat disimpulkan bahwa kinerja dua konfigurasi
model Catboost Classifier, yaitu model dengan parameter default dan model hasil hyperparameter
tuning, menunjukkan performa yang sangat baik dalam melakukan klasifikasi spesies penguin dengan model mencapai tingkat akurasi sebesar 99%. Hasil ini menunjukkan bahwa algoritma Catboost Classifier mampu menangkap pola klasifikasi dengan baik pada dataset berukuran kecil dan penerapan hyperparameter tuning dapat memberikan kontribusi dalam meningkatkan konsistensi performa model, serta penerapan Explainable Machine Learning berbasis SHAP berhasil memberikan interpretasi yang jelas terhadap pengaruh fitur morfologi dalam proses klasifikasi spesies penguin. Hal ini membuktikan bahwa penerapan SHAP mampu mengungkap kontribusi masing-masing fitur morfologi secara kuantitatif dan spesifik untuk setiap spesies, sehingga model Catboost yang digunakan tidak hanya bersifat akurat, tetapi juga dapat diinterpretasikan secara ilmiah.

## Daftar Pustaka
[A. Hua and G. Goldsztein, “Using Machine Learning to Predict Penguin Species,” J. Stud. Res., vol. 11, no. 4, Nov. 2022, doi: 10.47611/jsrhs.v11i4.3243] (https://doi.org/10.47611/jsrhs.v11i4.3243)

A. Pawar, “Data Analysis Using Statistical Methods: Case Study of Categorizing the Species of Penguin”.

[M. Zhu, J. Lai, X. Zhang, Y. Xu, and W. He, “An Explainable CatBoost Model for Crater Classification Based on Digital Elevation Model,” Remote Sens., vol. 17, no. 7, p. 1236, Mar. 2025, doi: 10.3390/rs17071236](https://doi.org/10.3390/rs17071236)

[N. Permatasari et al., “Predicting Diabetes Mellitus Using CatBoost Classifier and Shapley Additive Explanation (SHAP) Approach,” BAREKENG J. Ilmu Mat. Dan Terap., vol. 16, no. 2, pp. 615–624, 2022](https://doi.org/10.30598/barekengvol16iss2pp615-624)


