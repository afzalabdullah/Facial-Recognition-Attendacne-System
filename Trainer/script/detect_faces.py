import os
import cv2

face_cascade_path = './data/cascade/haarcascade_frontalface_default.xml'
images_path = './data/input_faces/'
faces_output_path = './data/output_crop_faces/'

def detect_faces():
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    
    if not os.path.exists(faces_output_path):
        os.makedirs(faces_output_path)
    
    for subdir, dirs, files in os.walk(images_path):
        for file in files:
            img_path = os.path.join(subdir, file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Unable to read image {img_path}")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
            
            if len(faces) == 0:
                print(f"No faces detected in {img_path}")
                continue

            for (x, y, w, h) in faces:
                face = img[y:y+h, x:x+w]
                folder_name = os.path.basename(subdir)
                output_folder = os.path.join(faces_output_path, folder_name)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                output_name = f"{os.path.splitext(file)[0]}_face_{x}_{y}_{w}_{h}.jpg"
                output_path = os.path.join(output_folder, output_name)
                cv2.imwrite(output_path, face)
                print(f"Done detecting: {output_path}")

if __name__ == "__main__":
    detect_faces()
