import pandas as pd
from sqlalchemy import create_engine
import time

# Tunggu Postgres siap
time.sleep(5)

# URL koneksi: postgresql://username:password@host:port/database
# Kita pakai host.docker.internal jika script jalan di luar docker (atau 'localhost' jika port ter-mapping)
engine = create_engine('postgresql://postgres:root@localhost:5433/students')

df = pd.read_csv('students_performance_processed.csv')

# Kirim data ke postgres
df.to_sql('students_performance', engine, if_exists='replace', index=False)
print("Berhasil mengunggah data CSV ke PostgreSQL!")
