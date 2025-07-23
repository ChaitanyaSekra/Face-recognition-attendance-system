from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
from deepface import DeepFace
from main import search_in_db, face_cascade

import psycopg2
from datetime import datetime, timedelta
from fastapi import Query

DB_CONFIG = {
    "dbname": "OpenCV",    
    "user": "postgres",            
    "password": "root",   
    "host": "localhost",         
    "port": 5432                   
}

# 🔁 Global frame buffer
latest_input_frame = None


def log_face_to_db(person_name):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        # Get the last log for this person
        cur.execute("""
            SELECT status, log_time FROM face_logs
            WHERE person_name = %s
            ORDER BY log_time DESC LIMIT 1
        """, (person_name,))
        last_entry = cur.fetchone()

        now = datetime.now()
        should_log = False
        next_status = "IN"  # default for first time

        if last_entry is None:
            should_log = True
        else:
            last_status, last_time = last_entry
            if (now - last_time) > timedelta(minutes=1):
                should_log = True
                next_status = "OUT" if last_status == "IN" else "IN"

        if should_log:
            cur.execute("""
                INSERT INTO face_logs (person_name, log_time, status)
                VALUES (%s, %s, %s)
            """, (person_name, now, next_status))
            conn.commit()
            print(f"[LOGGED] {person_name} - {next_status}")
        else:
            print(f"[SKIPPED] {person_name} (duplicate within 1 min)")

        cur.close()
        conn.close()

    except Exception as e:
        print("Logging DB error:", e)



app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ✅ Homepage route
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ✅ Client webcam page
@app.get("/client-cam", response_class=HTMLResponse)
async def client_cam(request: Request):
    return templates.TemplateResponse("client_cam.html", {"request": request})


# ✅ Receive webcam frame from client
@app.post("/client-frame/")
async def receive_client_frame(data: dict):
    global latest_input_frame

    try:
        img_data = data["image"].split(",")[1]
        img_bytes = base64.b64decode(img_data)
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        latest_input_frame = frame
        return {"status": "received"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ✅ Return live stream of recognized frames
@app.get("/output-stream")
def output_stream():
    def generate_output_frames():
        while True:
            if latest_input_frame is None:
                continue

            frame = latest_input_frame.copy()

            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                logged_this_frame = set()

                for (x, y, w, h) in faces:
                    face_crop = rgb_frame[y:y+h, x:x+w]

                    try:
                        embedding = DeepFace.represent(face_crop, model_name="ArcFace")[0]['embedding']
                        result = search_in_db(embedding)
                        label = result["Name"] if result else "Unknown"
                    except Exception as e:
                        print("Recognition error:", e)
                        label = "Unknown"

                    if label != "Unknown" and label not in logged_this_frame:
                        log_face_to_db(label)
                        logged_this_frame.add(label)

                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            except Exception as e:
                print("Processing error:", e)

            _, buffer = cv2.imencode(".jpg", frame)
            frame_bytes = buffer.tobytes()

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

    return StreamingResponse(generate_output_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/logs", response_class=HTMLResponse)
def logs_page(
    request: Request,
    person_name: str = Query(default=None),
    status: str = Query(default=None),
    date: str = Query(default=None),
    start_date: str = Query(default=None),
    end_date: str = Query(default=None)
):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        query = "SELECT person_name, log_time, status FROM face_logs WHERE 1=1"
        params = []

        if person_name:
            query += " AND person_name ILIKE %s"
            params.append(f"%{person_name}%")

        if status:
            query += " AND status = %s"
            params.append(status)

        if date:
            query += " AND DATE(log_time) = %s"
            params.append(date)

        if start_date and end_date:
            query += " AND DATE(log_time) BETWEEN %s AND %s"
            params.append(start_date)
            params.append(end_date)

        query += " ORDER BY log_time DESC"

        cur.execute(query, tuple(params))
        rows = cur.fetchall()

        logs = []
        for row in rows:
            logs.append({
                "person_name": row[0],
                "log_time": row[1].strftime("%Y-%m-%d %H:%M:%S"),
                "status": row[2]
            })

        cur.close()
        conn.close()

        filters = {
            "person_name": person_name,
            "status": status,
            "date": date,
            "start_date": start_date,
            "end_date": end_date
        }

        return templates.TemplateResponse("logs.html", {
            "request": request,
            "logs": logs,
            "filters": filters
        })

    except Exception as e:
        return HTMLResponse(f"<h2>Error loading logs: {e}</h2>", status_code=500)
    

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        cur.execute("""
            SELECT person_name, MAX(log_time) as latest, status FROM face_logs
            GROUP BY person_name, status
            HAVING status = 'IN'
        """)

        rows = cur.fetchall()
        people_inside = [
            {"name": row[0], "time": row[1].strftime("%Y-%m-%d %H:%M:%S")}
            for row in rows
        ]

        cur.close()
        conn.close()

        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "people": people_inside
        })

    except Exception as e:
        return HTMLResponse(f"<h2>Error loading dashboard: {e}</h2>", status_code=500)


# 🔒 Commented out (can re-enable when needed):
# from fastapi import UploadFile, File
# from fastapi.responses import FileResponse, RedirectResponse
# import shutil, os
# import datetime
# from main import add_new_face, recognize_faces, test_similarity_vs_distance, plot_score_vs_distance

# @app.get("/upload", response_class=HTMLResponse)
# async def upload_page(request: Request):
#     return templates.TemplateResponse("upload.html", {"request": request})

# @app.post("/upload", response_class=HTMLResponse)
# async def upload_face(request: Request, file: UploadFile = File(...)):
#     # handle file upload
#     pass

# @app.get("/add-face", response_class=HTMLResponse)
# async def add_face_page(request: Request):
#     add_new_face()
#     return templates.TemplateResponse("add_face.html", {"request": request, "msg": "✅ Faces added to DB."})

# @app.get("/recognize", response_class=HTMLResponse)
# async def recognize_page(request: Request):
#     recognize_faces()
#     return templates.TemplateResponse("recognize.html", {"request": request, "msg": "🧠 Face recognition session started."})

# @app.get("/analyze", response_class=HTMLResponse)
# async def analyze_page(request: Request):
#     distances, scores = test_similarity_vs_distance("distance_test/bhoomika")
#     plot_score_vs_distance(distances, scores)
#     return templates.TemplateResponse("analyze.html", {"request": request, "img_url": "/static/score_plot.png"})
