import http.client
import json
import uuid

def request(conn, method, url, data=None):
    headers = {"Content-Type": "application/json"} if data else {}
    conn.request(method, url, json.dumps(data) if data else None, headers)
    resp = conn.getresponse()
    body = resp.read().decode()
    return resp.status, json.loads(body) if body else None

def test_auth():
    print("Verifying Authentication Logic...")
    conn = http.client.HTTPConnection("localhost", 8000)
    
    # Use unique mobile for every test run
    unique_mobile_a = f"1{str(uuid.uuid4().int)[:9]}"
    unique_mobile_b = f"2{str(uuid.uuid4().int)[:9]}"

    # 1. Register User A
    user_a_data = {
        "mobile": unique_mobile_a,
        "password": "password123",
        "name": "Simba",
        "about": "The King"
    }
    status, body = request(conn, "POST", "/auth/register", user_a_data)
    print(f"Registration A: Status {status}, Body: {body}")
    
    if status != 200:
        print("FAIL: Registration A failed")
        return

    user_a_id = body['user_id']
    print(f"SUCCESS: User A registered with ID: {user_a_id}")

    # 2. Login User A
    login_data = {"mobile": unique_mobile_a, "password": "password123"}
    status, body = request(conn, "POST", "/auth/login", login_data)
    print(f"Login A: Status {status}, User: {body.get('name')}")
    
    # 3. Save History for User A
    history_data = {
        "name": "Simba",
        "result": {"happiness": "Ecstatic", "energy": "High"},
        "user_id": user_a_id
    }
    status, body = request(conn, "POST", "/history", history_data)
    print(f"Save History A: Status {status}")

    # 4. Filter History for User A
    status, history = request(conn, "GET", f"/history?user_id={user_a_id}")
    print(f"History A Count: {len(history)}")

    # 5. Register User B
    user_b_data = {
        "mobile": unique_mobile_b,
        "password": "password123",
        "name": "Scar",
        "about": "The Uncle"
    }
    status, body_b = request(conn, "POST", "/auth/register", user_b_data)
    user_b_id = body_b['user_id']
    
    # 6. Check History for User B (Should be empty)
    status, history_b = request(conn, "GET", f"/history?user_id={user_b_id}")
    print(f"History B Count (Should be 0): {len(history_b)}")

    if len(history_b) == 0:
        print("SUCCESS: Private History Filtering Verified!")
    else:
        print("FAIL: History Filtering Failed!")

    conn.close()

if __name__ == "__main__":
    test_auth()
