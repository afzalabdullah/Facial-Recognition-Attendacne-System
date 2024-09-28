import os
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from facenet_pytorch import InceptionResnetV1, MTCNN
from datetime import datetime, timedelta
import firebase_admin
from firebase_admin import credentials, storage
import mysql.connector
import tensorflow as tf
import csv
import schedule
import time

# Paths
arrays_path = './arrays/'
output_dir = './output/'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load saved embeddings and font size mappings
embeddings_data = np.load(os.path.join(arrays_path, 'embeddings.npz'))
slope_intercept_data = np.load(os.path.join(arrays_path, 'vars.npz'))

stored_embeddings = embeddings_data['a']
names = embeddings_data['b']
slope = slope_intercept_data['a']
intercept = slope_intercept_data['b']

# Load pre-trained MTCNN and Inception Resnet V1 models
device = 'cuda' if tf.config.list_physical_devices('GPU') else 'cpu'
mtcnn = MTCNN(keep_all=True, device=device, margin=20, post_process=True, min_face_size=40)
inception = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# MySQL connection details
db_config = {
    'user': 'root',
    'password': '',
    'host': 'localhost',
    'database': 'ams',
    'raise_on_warnings': True
}

# Initialize Firebase Admin SDK
cred = credentials.Certificate('./crediential.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'attendance-c8d63.appspot.com'
})

# Initialize the Firebase Storage bucket
bucket = storage.bucket()

def download_blob(blob, destination_file_name):
    """Downloads a blob from the bucket and saves it to a local file."""
    blob.download_to_filename(destination_file_name)
    print(f'Downloaded {blob.name} to {destination_file_name}')

def download_files():
    """Downloads files from the Firebase Storage bucket for the previous day's date."""
    previous_date_folder = (datetime.now() - timedelta(days=1)).strftime('%d%m%Y')
    destination_folder = os.path.join(os.getcwd(), previous_date_folder)
    
    # Check if the folder already exists
    if os.path.exists(destination_folder):
        print(f"Folder '{destination_folder}' already exists. Skipping download.")
        return destination_folder
    
    blobs = bucket.list_blobs(prefix=previous_date_folder + '/')
    os.makedirs(destination_folder, exist_ok=True)
    
    for blob in blobs:
        file_name = os.path.basename(blob.name)
        destination = os.path.join(destination_folder, file_name)
        download_blob(blob, destination)
    
    return destination_folder

def extract_face_embeddings(image):
    try:
        faces = mtcnn(image)
        if faces is None or len(faces) == 0:
            return np.array([])
        embeddings = inception(faces.to(device)).detach().cpu().numpy()
        return embeddings
    except Exception as e:
        print(f"Error in extract_face_embeddings: {e}")
        return np.array([])

def recognize_face(image, threshold=0.8):
    embeddings = extract_face_embeddings(image)
    if embeddings.size == 0:
        return "No face detected", None

    similarities = cosine_similarity(embeddings, stored_embeddings)
    best_match_index = np.argmax(similarities.mean(axis=0))
    best_score = similarities.mean(axis=0)[best_match_index]

    if best_score >= threshold:
        best_name = names[best_match_index]
    else:
        best_name = "Unidentified"

    return best_name, best_score

def process_image(image_path, date_folder):
    image = Image.open(image_path).convert('RGB')
    results = []

     # Extract timestamp from the image filename
    image_filename = os.path.basename(image_path)
    # Assuming the filename contains the timestamp, e.g., 'checkin_20230820_083000.png'
    timestamp_str = image_filename.split('_')[1].split('.')[0]

    try:
        faces, _ = mtcnn.detect(image)
        print(f"Detected faces: {faces}")
    except Exception as e:
        print(f"Error in face detection: {e}")
        faces = None

    if faces is not None:
        for idx, box in enumerate(faces):
            try:
                x_face, y_face, x_face_end, y_face_end = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                face_image = image.crop((x_face, y_face, x_face_end, y_face_end))
                
                recognized_name, confidence = recognize_face(face_image)
                if recognized_name not in ["No face detected", "Unidentified"]:
                    results.append(recognized_name)
                    
                    # Create directories for check-in and check-out
                    output_date_folder = os.path.join(output_dir, date_folder)
                    checkin_dir = os.path.join(output_date_folder, 'checkin')
                    checkout_dir = os.path.join(output_date_folder, 'checkout')
                    
                    os.makedirs(checkin_dir, exist_ok=True)
                    os.makedirs(checkout_dir, exist_ok=True)
                    
                    # Determine if the image is check-in or check-out
                    is_checkin = 'checkin' in image_path
                    is_checkout = 'checkout' in image_path
                    
                    if is_checkin:
                        face_filename = f"{recognized_name}_{idx}_{timestamp_str}.png"
                        face_image.save(os.path.join(checkin_dir, face_filename))
                    elif is_checkout:
                        face_filename = f"{recognized_name}_{idx}_{timestamp_str}.png"
                        face_image.save(os.path.join(checkout_dir, face_filename))
                        
            except Exception as e:
                print(f"Error processing face: {e}")
                continue

    return results

def extract_info_from_filename(filename):
    parts = filename.split('_')
    date = parts[1]
    time = parts[2].split('.')[0]
    
    date_formatted = datetime.strptime(date, '%d%m%Y').strftime('%Y-%m-%d')
    time_formatted = datetime.strptime(time, '%H%M%S').strftime('%H:%M:%S')
    
    return date_formatted, time_formatted

def upsert_attendance_record(name, date, checkin_time=None, checkout_time=None):
    try:
        table_name = "attendance"
        cnx = mysql.connector.connect(**db_config)
        cursor = cnx.cursor()

        name = str(name)
        cnx.start_transaction()

        # Fetch existing record
        query = (f"SELECT id, checkin_time, checkout_time FROM {table_name} "
                 "WHERE name = %s AND date = %s")
        cursor.execute(query, (name, date))
        result = cursor.fetchone()

        if result:
            record_id, existing_checkin, existing_checkout = result

            # Update checkin_time if not set and checkin_time is provided
            if existing_checkin is None and checkin_time:
                # if name in ['5719', '5852', '5895']:
                #     checkin_time = adjust_time(checkin_time, 25)
                update_checkin = (f"UPDATE {table_name} "
                                  "SET checkin_time = %s "
                                  "WHERE id = %s")
                cursor.execute(update_checkin, (checkin_time, record_id))

            # Update checkout_time if not set and checkout_time is provided
            if existing_checkout is None and checkout_time:
                
                update_checkout = (f"UPDATE {table_name} "
                                   "SET checkout_time = %s "
                                   "WHERE id = %s")
                cursor.execute(update_checkout, (checkout_time, record_id))

        else:
            # For new records, adjust checkin_time if needed
            # if name in ['5719', '5852', '5895']:
            #     if checkin_time is None:
                    # checkin_time = '090000'  # Default checkin time in HHMMSS format
                # checkin_time = adjust_time(checkin_time, 25)

            # Insert new record
            add_record = (f"INSERT INTO {table_name} "
                          "(name, date, checkin_time, checkout_time) "
                          "VALUES (%s, %s, %s, %s)")
            cursor.execute(add_record, (name, date, checkin_time, checkout_time))

        cnx.commit()

    except mysql.connector.Error as err:
        print(f"Error: {err}")
        if cnx.is_connected():
            cnx.rollback()
    finally:
        if cursor:
            cursor.close()
        if cnx:
            cnx.close()

def main():
    folder = download_files()
    current_date = datetime.now().strftime('%Y%m%d')
    csv_file = f'recognition_results_{current_date}.csv'

    date_folder = os.path.basename(folder)

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Name', 'Date', 'Checkin Time', 'Checkout Time'])

        for image_name in os.listdir(folder):
            if image_name.endswith('.jpg') or image_name.endswith('.png'):
                image_path = os.path.join(folder, image_name)
                recognized_names = process_image(image_path, date_folder)
                date, time = extract_info_from_filename(image_name)
                
                checkin_time = time if 'checkin' in image_name else None
                checkout_time = time if 'checkout' in image_name else None
                
                for name in recognized_names:
                    writer.writerow([name, date, checkin_time, checkout_time])
                    upsert_attendance_record(name, date, checkin_time, checkout_time)

    print(f"Results saved to {csv_file}")
    print(f"Detected face images saved to {output_dir}")

# Schedule the task to run daily at 10 AM
schedule.every().day.at("10:00").do(main)

while True:
    schedule.run_pending()
    time.sleep(1)
