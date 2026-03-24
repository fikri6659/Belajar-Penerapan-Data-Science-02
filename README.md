# Proyek Akhir: Menyelesaikan Permasalahan Institusi Pendidikan

## Business Understanding

Jaya Jaya Institut merupakan institusi pendidikan yang berdiri sejak tahun 2000 dan telah mencetak banyak lulusan dengan reputasi yang sangat baik. Akan tetapi, terdapat banyak juga siswa yang tidak menyelesaikan pendidikannya alias *dropout*.

### Permasalahan Bisnis

Tingginya jumlah *dropout* tentunya menjadi salah satu masalah yang besar untuk institusi pendidikan. Oleh karena itu, Jaya Jaya Institut ingin mendeteksi secepat mungkin siswa yang mungkin akan melakukan *dropout* sehingga dapat diberikan bimbingan khusus atau intervensi dini sebelum terlambat.

### Cakupan Proyek

- Melakukan pembersihan dan eksplorasi data (EDA) pada dataset performa siswa dari Dicoding Academy.
- Menganalisis faktor-faktor yang paling memengaruhi risiko *dropout* siswa.
- Mengembangkan model *machine learning* klasifikasi untuk mendeteksi risiko dini *dropout*.
- Membuat *Business Dashboard* interaktif untuk memudahkan pemantauan staf institusi.
- Menyediakan web prototipe dengan Streamlit bagi pihak institusi agar cepat melakukan *screening*.

### Persiapan Lingkungan dan Cara Menjalankan

Sumber data yang digunakan dapat diunduh pada repositori: [students&#39; performance](https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/README.md) dataset.

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

> Aplikasi secara responsif akan berjalan di jendela browser pada `http://localhost:8501`. (Jika finalisasi telah diunggah / didistribusikan ke server *Streamlit Community Cloud*, tambahkan dan arahkan url *reviewer* ke ( https://jsd8dcdn4vftyq77u6sf5w.streamlit.app/ ).

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

## Dataset yang Digunakan

Dataset yang digunakan dalam proyek ini adalah **Students Performance Dataset** dari Dicoding Academy yang berisi informasi tentang **4,424 siswa perguruan tinggi** dengan atribut sebagai berikut:

### Atribut Utama:

- **Marital_status**: Status pernikahan orang tua (1=Single, 2=Other)
- **Application_mode**: Mode aplikasi pendaftaran (1-42)
- **Application_order**: Urutan aplikasi (1-8)
- **Course**: Kode program studi (171, 9254, 9070, dll)
- **Daytime_evening_attendance**: Kehadiran siang/malam (1=Siang, 0=Malam)
- **Previous_qualification**: Kualifikasi sebelumnya (1-42)
- **Previous_qualification_grade**: Nilai kualifikasi sebelumnya (0-200)
- **Nacionality**: Kebangsaan (1-43)
- **Mothers_qualification**: Kualifikasi ibu (1-43)
- **Fathers_qualification**: Kualifikasi ayah (1-43)
- **Mothers_occupation**: Pekerjaan ibu (0-9)
- **Fathers_occupation**: Pekerjaan ayah (0-9)
- **Admission_grade**: Nilai admission (0-200)
- **Displaced**: Apakah dipindahkan (1=Ya, 0=Tidak)
- **Educational_special_needs**: Kebutuhan khusus (1=Ya, 0=Tidak)
- **Debtor**: Apakah debitor (1=Ya, 0=Tidak)
- **Tuition_fees_up_to_date**: Biaya kuliah up to date (1=Ya, 0=Tidak)
- **Gender**: Jenis kelamin (1=Male, 0=Female)
- **Scholarship_holder**: Apakah beasiswa (1=Ya, 0=Tidak)
- **Age_at_enrollment**: Usia saat enrollment (16-70)
- **International**: Apakah internasional (1=Ya, 0=Tidak)
- **Curricular_units_1st_sem**: Unit kurikuler semester 1 (credited, enrolled, evaluations, approved, grade, without_evaluations)
- **Curricular_units_2nd_sem**: Unit kurikuler semester 2 (credited, enrolled, evaluations, approved, grade, without_evaluations)
- **Unemployment_rate**: Tingkat pengangguran (0-20%)
- **Inflation_rate**: Tingkat inflasi (-5% sampai 10%)
- **GDP**: Produk Domestik Bruto (-5% sampai 10%)

### Target Variable:

- **Status**: Status siswa (Dropout, Enrolled, Graduate)

## Catatan Spesifik Evaluasi Reviewer

Bagi Reviewer penilai yang hanya perlu mengecek dan mengevaluasi status akhir desain grafis dari **Metabase** tanpa me-*run* kembali konfigurasi panjang *Docker*-nya dari nol, kami telah menyediakan cetakan *database environment* (*snapshot*) di ekstensi format statis **`metabase.db.mv.db`** (Terletak di *root project* berkas arsip ZIP saat ini). Anda dapat memuatnya ulang dengan aman tanpa cemas relasi pangkalan data terhapus.

Sementara itu, dari segi kinerja prediksi deteksi sistem Machine Learning, aplikasi utama di atas ditenagai oleh skema pemodelan kompleks *Random Forest Classifier*, dan berhasil mencapai kepekaan prediksi (*Accuracy*) **76.84%** serta pemisahan ROC AUC Score di rasio yang sangat baik yaitu **0.885**. Sangat memadai untuk penggunaan komersil penanggulan *dropout* institut nyata.

## Conclusion

Dari hasil pelatihan machine learning dan analisis Exploratory Data Analysis (EDA), dapat disimpulkan bahwa:

1. **Faktor Akademik Dominan**: Nilai akademis siswa, terutama pada semester kedua (approved dan grade), merupakan prediktor terkuat untuk status siswa.
2. **Faktor Administratif**: Status biaya kuliah (tuition_fees_up_to_date), beasiswa (scholarship_holder), dan admission grade memiliki pengaruh signifikan.
3. **Faktor Demografis**: Usia saat enrollment dan status internasional juga mempengaruhi kelangsungan studi siswa.
4. **Faktor Ekonomi**: Tingkat pengangguran dan inflasi di lingkungan siswa berpengaruh terhadap kemampuan mereka menyelesaikan studi.

### Rekomendasi Action Items

Berdasarkan hasil analisis, berikut adalah rekomendasi untuk Jaya Jaya Institut:

1. **Intervensi Akademik Dini**: Identifikasi siswa dengan nilai rendah pada semester pertama dan berikan bimbingan akademik intensif.
2. **Dukungan Administratif**: Perbaiki sistem pembayaran biaya kuliah dan berikan insentif bagi siswa yang menjaga status pembayaran aktif.
3. **Program Beasiswa**: Perluas program beasiswa bagi siswa berprestasi dan siswa kurang mampu.
4. **Pembimbingan Karier**: Berikan pembimbingan karier sejak awal untuk membantu siswa memahami komitmen yang diperlukan.
5. **Monitoring Real-time**: Implementasikan sistem monitoring untuk staf akademik agar dapat mengidentifikasi siswa berisiko sejak dini.
