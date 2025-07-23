import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from datetime import datetime
import csv
from deepface import DeepFace
from pinecone import Pinecone
import matplotlib.pyplot as plt
import shutil


#Api_key = os.environ.get("pcsk_2J7EVG_627jkBdNjsMmk9HQrSgrDLet68txQzv3R4dvy4kWePXSQ3tLcWL2emAg2f9iKjf")
#Environment = os.environ.get("us-east-1")
pc = Pinecone(api_key="your_api_rey", environment="region")

DB_PATH = "face_database"
FACES_ADDED_PATH = "faces_added"

# Load Haarcascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

if not os.path.exists(DB_PATH):
    os.makedirs(DB_PATH)

# Initialize face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

def insert_into_db(embedding_dict):

    index_name = "your_index_name"
    if not pc.has_index(index_name):
        pc.create_index_for_model(
            name=index_name,
            cloud="aws",
            region="us-east-1",
            embed={
                "model":"llama-text-embed-v2",
                "field_map":{"text": "chunk_text"}
            }
        )

    dense_index = pc.Index(index_name)
    dense_index.upsert(vectors=embedding_dict, namespace="__default__")


def preprocess_image(img_path, target_size=(112, 112)): 
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image not found or unreadable: {img_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(gray)

    #Denoising
    #denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # Histogram equalization to enhance contrast
    equalized = cv2.equalizeHist(contrast_enhanced)

    # Sharpening kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    
    sharpened = cv2.filter2D(equalized, -1, kernel)

    # Resize to target size
    resized = cv2.resize(sharpened, (112,112))

    gamma = 1.2
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    gamma_corrected = cv2.LUT(equalized, table)

    # Convert back to 3-channel RGB (DeepFace expects color)
    final_img = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    final_img = cv2.cvtColor(gamma_corrected, cv2.COLOR_GRAY2RGB)

    return final_img


# Function to capture images and save them to database
def add_new_face():
    embedding_list = []

    for person_name in os.listdir(DB_PATH):
        person_dir = os.path.join(DB_PATH, person_name)
        if not os.path.isdir(person_dir):
            continue

        print(f"\nProcessing images for: {person_name}")
        count = 0

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)

            img = cv2.imread(img_path)
            if img is None:
                print(f"❌ Failed to read {img_path}")
                continue

            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            try:
                # Get face embedding
                embedding = DeepFace.represent(rgb_img, model_name="ArcFace")
                embedding_vector = embedding[0]["embedding"]

                embedding_dict = {
                    "id": f"{person_name}_{count}",
                    "values": embedding_vector,
                    "metadata": {
                        "Name": person_name
                    }
                }
                embedding_list.append(embedding_dict)
                print(f"✔️ Embedded {img_path} -> ID: {person_name}_{count}")
                count += 1

                # Move image to faces_added/{person_name}/
                dest_dir = os.path.join(FACES_ADDED_PATH, person_name)
                os.makedirs(dest_dir, exist_ok=True)
                dest_path = os.path.join(dest_dir, img_name)
                shutil.move(img_path, dest_path)
                print(f"📦 Moved to: {dest_path}")
                if embedding_list:
                    insert_into_db(embedding_list)
                    print(f"\n✅ Inserted {len(embedding_list)} embeddings into database.")
                else:
                    print("⚠️ No valid images found. Nothing inserted.")
            except Exception as e:
                print(f"❌ Error processing {img_path}: {e}")
                # Image stays in original folder if it fails

    return

def search_in_db(query_embedding, top_k=1, threshold=0.5):
    index_name = "frs"
    index = pc.Index(index_name)

    try:
        response = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

        if response and response['matches']:
            top_match = response['matches'][0]
            if top_match['score'] >= threshold:
                return top_match['metadata']  # e.g., {"Name": "Alice"}
            else:
                print(f"⚠️ Match found but below threshold: {top_match['score']:.4f}")
                return None
        else:
            return None

    except Exception as e:
        print(f"❌ Pinecone query error: {e}")
        return None
# Function to recognize faces
def recognize_faces():
    cap = cv2.VideoCapture(0)
    print("🔍 Starting face recognition... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

        for (x, y, w, h) in faces:
            face_crop = rgb_frame[y:y + h, x:x + w]

            try:
                embedding = DeepFace.represent(face_crop, model_name="ArcFace")[0]['embedding']
                result = search_in_db(embedding)

                label = result['Name'] if result else "Unknown"

            except Exception as e:
                label = "Error"

            # Draw bounding box + label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Live Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def test_similarity_vs_distance(folder_path):
    distances = []
    scores = []
    index_name = "frs"
    index = pc.Index(index_name)

    for img_name in sorted(os.listdir(folder_path), key=lambda x: int(os.path.splitext(x)[0])):
        img_path = os.path.join(folder_path, img_name)
        distance = int(os.path.splitext(img_name)[0])  # e.g., "2.jpg" → 2 feet

        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Failed to load {img_path}")
            continue

        try:
            #rgb_img = preprocess_image(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            embedding = DeepFace.represent(rgb_img, model_name="ArcFace")[0]['embedding']

            # Query Pinecone for similarity
            response = index.query(
                vector=embedding,
                top_k=1,
                include_metadata=True
            )

            if response['matches']:
                score = response['matches'][0]['score']
                distances.append(distance)
                scores.append(score)
                print(f"{img_name}: {score:.4f}")
            else:
                print(f"No match for {img_name}")
        except Exception as e:
            print(f"Error with {img_name}: {e}")

    return distances, scores


def plot_score_vs_distance(distances, scores):
    plt.figure(figsize=(8, 5))
    plt.plot(distances, scores, marker='o', color='blue', linestyle='-')
    plt.title("Similarity Score vs Distance for Chaitanya's Face")
    plt.xlabel("Distance (feet)")
    plt.ylabel("Cosine Similarity Score")
    plt.grid(True)
    plt.tight_layout()
    #plt.show()
    plt.savefig("static/score_plot.png")


# Main menu
def main():
    while True:
        #distances, scores = test_similarity_vs_distance("distance_test/chaitanya")
        #plot_score_vs_distance(distances, scores)
        #recognize_faces()
        add_new_face()
        break

if __name__ == "__main__":
    main()
