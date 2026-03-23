import requests
import json
import time

HOST = "http://localhost:3001"
HEADERS = {}

def login():
    r = requests.post(f"{HOST}/api/session", json={
        "username": "root@mail.com",
        "password": "root123"
    })
    HEADERS["X-Metabase-Session"] = r.json()["id"]

def get_db_id():
    dbs = requests.get(f"{HOST}/api/database", headers=HEADERS).json()
    for d in dbs.get("data", []):
        if d.get("name") == "Students PostgreSQL DB":
            return d["id"]
    return None

def create_card(name, query, db_id, display="table"):
    payload = {
        "name": name,
        "dataset_query": {
            "type": "native",
            "native": {"query": query},
            "database": db_id
        },
        "display": display,
        "visualization_settings": {}
    }
    r = requests.post(f"{HOST}/api/card", headers=HEADERS, json=payload)
    if r.status_code == 200:
        print(f"Card '{name}' dibuat, ID: {r.json()['id']}")
        return r.json()["id"]
    else:
        print(f"Gagal buat card '{name}':", r.text)
        return None

def create_dashboard(name):
    payload = {"name": name, "parameters": []}
    r = requests.post(f"{HOST}/api/dashboard", headers=HEADERS, json=payload)
    if r.status_code == 200:
        print(f"Dashboard '{name}' dibuat, ID: {r.json()['id']}")
        return r.json()["id"]
    return None

def add_card_to_dash(dash_id, card_id, row, col, size_x, size_y):
    payload = {
        "cardId": card_id,
        "row": row,
        "col": col,
        "size_x": size_x,
        "size_y": size_y
    }
    r = requests.post(f"{HOST}/api/dashboard/{dash_id}/cards", headers=HEADERS, json=payload)
    if r.status_code == 200:
        print(f"Card {card_id} ditambahkan ke dash {dash_id}")

def main():
    try:
        login()
    except Exception as e:
        print("Belum bisa login", e)
        return

    db_id = get_db_id()
    if not db_id:
        print("DB tidak ditemukan")
        return
        
    c1 = create_card("Risk Count", "SELECT CASE WHEN dropout_risk = 1 THEN 'Berisiko' ELSE 'Aman' END AS status, COUNT(*) FROM students_performance GROUP BY dropout_risk", db_id, "pie")
    c2 = create_card("Average Math by Lunch", "SELECT lunch, AVG(\"math score\") AS avg_math FROM students_performance GROUP BY lunch", db_id, "bar")
    c3 = create_card("High Risk Students", "SELECT * FROM students_performance WHERE dropout_risk = 1 ORDER BY average_score ASC LIMIT 50", db_id, "table")
    
    dash_id = create_dashboard("Jaya Jaya Dashboard Baru")
    if dash_id:
        if c1: add_card_to_dash(dash_id, c1, 0, 0, 6, 6)
        if c2: add_card_to_dash(dash_id, c2, 0, 6, 6, 6)
        if c3: add_card_to_dash(dash_id, c3, 6, 0, 12, 6)
        
        # publish dashboard logic exists internally, it's viewable by admins.
        
    print("Dashboard pembuatan selesai!")

if __name__ == "__main__":
    main()
