import requests
import json
import time

HOST = "http://localhost:3001"

def wait_for_metabase():
    print("Menunggu Metabase menyala (bisa beberapa menit)...")
    while True:
        try:
            r = requests.get(f"{HOST}/api/health")
            if r.status_code == 200:
                props = requests.get(f"{HOST}/api/session/properties").json()
                if "setup-token" in props and props["setup-token"]:
                    print("Metabase sedia untuk setup!")
                    return props["setup-token"]
                elif "setup-token" not in props:
                    print("Metabase sudah disetup.")
                    return None
        except:
            pass
        time.sleep(5)

def setup_metabase(setup_token):
    print("Menjalankan initial setup Metabase...")
    payload = {
        "token": setup_token,
        "user": {
            "first_name": "Admin",
            "last_name": "User",
            "email": "root@mail.com",
            "password": "root123",
            "site_name": "Jaya Jaya Institut Dashboard"
        },
        "prefs": {
            "site_name": "Jaya Jaya Institut Dashboard",
            "allow_tracking": False
        },
        "database": None
    }
    r = requests.post(f"{HOST}/api/setup", json=payload)
    if r.status_code == 200:
        print("Setup berhasil!")
        return r.json().get("id")
    else:
        print("Setup gagal:", r.text)
        return None

def login():
    r = requests.post(f"{HOST}/api/session", json={
        "username": "root@mail.com",
        "password": "root123"
    })
    return r.json()["id"]

def add_database(session_id):
    print("Menambah PostgreSQL sebagai sumber data...")
    headers = {"X-Metabase-Session": session_id}
    # Perhatikan host.docker.internal karena metabase dijalankan via docker
    payload = {
        "engine": "postgres",
        "name": "Students PostgreSQL DB",
        "details": {
            "host": "host.docker.internal",
            "port": 5433,
            "dbname": "students",
            "user": "postgres",
            "password": "root"
        },
        "is_full_sync": True,
        "is_on_demand": False
    }
    r = requests.post(f"{HOST}/api/database", headers=headers, json=payload)
    if r.status_code == 200:
        print("Database berhasil ditambahkan!")
        return r.json()["id"]
    else:
        print("Gagal menambahkan database:", r.text)
        return None

def force_sync(session_id, db_id):
    headers = {"X-Metabase-Session": session_id}
    requests.post(f"{HOST}/api/database/{db_id}/sync_schema", headers=headers)
    print("Menunggu sync skema...")
    time.sleep(10)

def main():
    token = wait_for_metabase()
    if token:
        setup_metabase(token)
    
    session_id = login()
    print("Berhasil login ke Metabase API.")
    
    # Cek apakah db sudah ada
    headers = {"X-Metabase-Session": session_id}
    dbs = requests.get(f"{HOST}/api/database", headers=headers).json()
    db_id = None
    for d in dbs.get("data", []):
        if d.get("name") == "Students PostgreSQL DB":
            db_id = d["id"]
            break
            
    if not db_id:
        db_id = add_database(session_id)
        force_sync(session_id, db_id)
    
    print("Semua setup berhasil. Silakan buka http://localhost:3001")

if __name__ == "__main__":
    main()
