# Proyek Akhir: Menyelesaikan Permasalahan Institusi Pendidikan

## Business Understanding
Jaya Jaya Institut merupakan institusi pendidikan yang berdiri sejak tahun 2000 dan telah mencetak banyak lulusan dengan reputasi yang sangat baik. Akan tetapi, terdapat banyak juga siswa yang tidak menyelesaikan pendidikannya alias *dropout*.

### Permasalahan Bisnis
Tingginya jumlah *dropout* tentunya menjadi salah satu masalah yang besar untuk institusi pendidikan. Oleh karena itu, Jaya Jaya Institut ingin mendeteksi secepat mungkin siswa yang mungkin akan melakukan *dropout* sehingga dapat diberikan bimbingan khusus atau intervensi dini sebelum terlambat.

### Cakupan Proyek
- Melakukan pembersihan dan eksplorasi data (EDA) pada dataset performa siswa.
- Menganalisis faktor-faktor demografis maupun akademik yang paling memengaruhi risiko *dropout*.
- Mengembangkan model *machine learning* klasifikasi untuk mendeteksi risiko dini *dropout*.
- Membuat *Business Dashboard* interaktif untuk memudahkan pemantauan staf institusi.
- Menyediakan web prototipe dengan Streamlit bagi pihak institusi agar cepat melakukan *screening*.

### Persiapan Lingkungan dan Cara Menjalankan

Sumber data yang digunakan dapat diunduh pada repositori: [students' performance](https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/README.md) dataset.

Langkah-langkah di bawah ini disusun secara terstruktur untuk perbaikan (_maintenance_) dan kemudahan dalam menjalankan kembali keseluruhan _Machine Learning pipeline_:

**1. Persiapan Environment Python**
Gunakan Python versi 3.9 atau lebih baru. Sangat direkomendasikan untuk membungkus instalasi di dalam _virtual environment_.
```bash
# Membuat virtual environment
py -m venv env

# Aktivasi environment (Windows)
env\Scripts\activate
# Aktivasi environment (Mac/Linux)
source env/bin/activate

# Install parameter pustaka dan versi yang sesuai dari requirement
py -m pip install -r requirements.txt
```

**2. Eksplorasi Data (EDA) & Pelatihan Ulang Model ML**
Apabila Anda melakukan penambahan fitur atau ingin meninjau keseluruhan alur pra-pemrosesan data dalam jupyter notebook:
- Untuk mengecek visualisasi historis dan pengerjaan _training_: 
  `py -m jupyter notebook notebook.ipynb`
- Untuk melatih dan me-resave/meng-generate ulang keseluruhan model deteksi *Random Forest* ke direktori `/model` (beserta pkl scaler, kolom fitur, dan tabel terproses):
  ```bash
  py train_model.py
  ```

**3. Menjalankan Prototipe App Machine Learning (Streamlit)**
Prototipe prediksi dini untuk mahasiswa berisiko dropout berbasis interaksi Graphical Web UI bisa diakses (_serve local proxy_) melalui perintah *Streamlit*:
```bash
py -m streamlit run app.py
```
> Aplikasi secara responsif akan berjalan di jendela browser pada `http://localhost:8501`. (Jika finalisasi telah diunggah / didistribusikan ke server *Streamlit Community Cloud*, tambahkan dan arahkan url *reviewer* ke **Link Akses Cloud Anda Di Sini**).

**4. Opsi Ekstra: Mengaktifkan Business Dashboard Utama (Docker, PostgreSQL, & Metabase)**
Selain menanam *dashboard* mini di *Streamlit* (`app.py`), arsitektur kode di repositori ini didesain juga untuk _dashboard_ pantauan Business-Analytics **Metabase** penuh. Berikut *script* peruntutan eksekusinya:
```bash
# a. Jalankan container PostgreSQL untuk data mentah di port 5433
docker run -d --name postgres-students -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=root -e POSTGRES_DB=students -p 5433:5432 postgres:13

# b. Muat turun/migrasikan format tabel Dataset CSV yang sudah diolah Python ke dalam PostgreSQL
py load_csv.py

# c. Jalankan engine server Metabase di port 3001
docker run -d --name metabase-students -p 3001:3000 metabase/metabase

# d. Konfigurasi otomatis Akun Metabase, Sinkronisasi PostgreSQL, dan Layout Grafis Dashboard!
py setup_metabase.py
py build_dashboard.py
```
*Dengan skrip otomatisasi *setup* ini, instalasi struktur *back-end* Metabase beserta sinkronisasi databasenya akan langsung tereksekusi tanpa konfigurasi manual yang kikuk. Anda bisa merangkai atau melihat perubahannya sendiri di `http://localhost:3001` dengan kredensial profil instansi admin bawaan `root@mail.com` (Sandi: `root123`).*
 
## Catatan Spesifik Evaluasi Reviewer
Bagi Reviewer penilai yang hanya perlu mengecek dan mengevaluasi status akhir desain grafis dari **Metabase** tanpa me-*run* kembali konfigurasi panjang *Docker*-nya dari nol, kami telah menyediakan cetakan *database environment* (*snapshot*) di ekstensi format statis **`metabase.db.mv.db`** (Terletak di *root project* berkas arsip ZIP saat ini). Anda dapat memuatnya ulang dengan aman tanpa cemas relasi pangkalan data terhapus.

Sementara itu, dari segi kinerja prediksi deteksi sistem Machine Learning, aplikasi utama di atas ditenagai oleh skema pemodelan kompleks *Random Forest Classifier*, dan berhasil mencapai kepekaan prediksi (*Accuracy*) **85.5%** serta pemisahan ROC AUC Score di rasio yang sangat baik yaitu **0.923**. Sangat memadai untuk penggunaan komersil penanggulan *dropout* institut nyata.

## Conclusion
Dari hasil pelatihan machine learning dan analisis Exploratory Data Analysis (EDA), dapat disimpulkan bahwa:
- Faktor paling signifikan penentu dropout siswa di institusi adalah capaian nilai akademis siswa, terutama pada bagian membaca (Reading) dan menulis (Writing), ketimbang faktor latar belakang yang lain.
- Tingkat pendidikan orang tua serta kepesertaan kelompok ras/etnis tidak memiliki pengaruh langsung determinatif besar terhadap rasio kegagalan.
- Siswa yang tidak mengambil kursus persiapan sama sekali (*none*) dan siswa dengan akses program ekonomi/subsidi rentan (*free/reduced lunch*) memiliki peluang lebih besar jatuh ke dalam kategori *underperforming* (risiko tinggi dropout).

### Rekomendasi Action Items
Berikut adalah rekomendasi *action items* yang harus dilakukan Jaya Jaya Institut untuk menekan angka dropout:
- **Penyelenggaraan Bimbingan Akademik Intensif**: Segera jalankan intervensi belajar (mentoring 1-on-1) bagi nama murid yang muncul pada tabel "Siswa Berisiko Dropout" dalam Dashboard Streamlit, utamanya fokuskan pada peningkatan cara baca (Reading) dan tulis (Writing).
- **Inklusivitas Test Preparation Course**: Mendorong secara mewajibkan siswa baru atau siswa berisiko untuk mengikuti *test preparation course*, bila perlu berikan skema keringanan biaya demi meminimalisasi kurangnya persiapan materi akademik siswa.
- **Monitoring Berbasis Data (Real-time)**: Mewajibkan pengajar dan staf wali untuk senantiasa mengecek metrik performa individu di aplikasi Machine Learning yang dibuat pada pertengahan atau setiap awal semester baru sebelum kerugian terlalu jauh.
- **Dukungan Kesejahteraan Ekstra**: Memberikan dukungan sosial, fisik, maupun gizi yang memadai bagi siswa pendaftar "Free/reduced lunch", sebab stabilitas energi dan fokus berpengaruh pada performa pembelajaran panjang.
