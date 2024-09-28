import os
import numpy as np
from PIL import Image, ImageFont
from facenet_pytorch import MTCNN, InceptionResnetV1
from detect_faces import detect_faces
from sklearn.linear_model import LinearRegression
import tensorflow as tf

# Paths
faces_path = './data/output_crop_faces/'
images_path = './data/input_faces/'
font_path = './data/font/Calibri Regular.ttf'
arrays_path = './data/arrays/'

# Load pre-trained MTCNN and Inception Resnet V1 models
device = 'cuda' if tf.config.list_physical_devices('GPU') else 'cpu'
mtcnn = MTCNN(keep_all=True, device=device)
inception = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def extract_face_embeddings(image_path):
    # Load image
    img = Image.open(image_path)
    
    # Detect faces
    faces = mtcnn(img)
    
    if faces is None:
        return np.array([])
    
    # Extract embeddings
    embeddings = inception(faces.to(device)).detach().cpu().numpy()
    return embeddings

def load_existing_data():
    try:
        with np.load(os.path.join(arrays_path, 'embeddings.npz')) as data:
            existing_faces = data['a']
            existing_names = data['b']
        
        with np.load(os.path.join(arrays_path, 'vars.npz')) as data:
            existing_slope = data['a']
            existing_intercept = data['b']
        
        return existing_faces, existing_names, existing_slope, existing_intercept
    except FileNotFoundError:
        return np.array([]), np.array([]), np.array([]), np.array([])

def update_embeddings(new_faces, new_names, existing_faces, existing_names):
    if len(existing_faces) == 0:
        return new_faces, new_names
    
    # Concatenate existing and new faces
    all_faces = np.vstack((existing_faces, new_faces))
    all_names = np.concatenate((existing_names, new_names))
    
    # Remove duplicates based on names
    unique_names, indices = np.unique(all_names, return_index=True)
    unique_faces = all_faces[indices]
    
    return unique_faces, unique_names

def update_font_size_mappings(new_names, existing_names, existing_slope, existing_intercept):
    # Combine existing and new names
    combined_names = np.concatenate((existing_names, new_names))
    slopes = np.zeros(len(combined_names))
    intercepts = np.zeros(len(combined_names))
    
    # Calculate slopes and intercepts for new names
    for name in combined_names:
        x = []
        y = []
        for j in range(1, 100):
            font = ImageFont.truetype(font_path, j)
            bbox = font.getbbox(name)
            x.append(j)
            y.append(bbox[2])
        lin = LinearRegression().fit(np.array(y).reshape(-1, 1), np.array(x))
        slopes[np.where(combined_names == name)[0]] = lin.coef_[0]
        intercepts[np.where(combined_names == name)[0]] = lin.intercept_
    
    return slopes, intercepts

# Choice for face detection
choice = int(input("""Choose one:
        1. Have faces already.
        2. Want to detect faces from pictures in './data/images/'\n"""))
if choice == 2:
    detect_faces()
    print("\n\n\nFaces saved in './data/faces/'\n\n\n")
    print("\n\n\nNOTE: PLEASE DELETE UNNECESSARY FACES BEFORE PROCEEDING AND RENAMING FACES TO THEIR NAMES.\n\n")
    input("Press ENTER to continue.(IF YOU'VE DELETED UNNECESSARY FACES AND RENAMED FACES)\n")
elif choice != 1:
    print('Wrong Choice')
    input()
    quit()

# Load existing data
existing_faces, existing_names, existing_slope, existing_intercept = load_existing_data()

# Get face images and names
new_faces = []
new_names = []

for root, dirs, files in os.walk(faces_path):
    for file in files:
        if file.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(root, file)
            embeddings = extract_face_embeddings(img_path)
            
            if embeddings.size > 0:
                new_faces.append(embeddings)
                new_names.append(os.path.basename(root))

if len(new_faces) == 0:
    print("No new Face Found")
    input()
    quit()

new_faces = np.vstack(new_faces)  # Stack all embeddings into a single array
new_names = np.array(new_names)

# Update embeddings
updated_faces, updated_names = update_embeddings(new_faces, new_names, existing_faces, existing_names)

# Update font size mappings
updated_slope, updated_intercept = update_font_size_mappings(new_names, existing_names, existing_slope, existing_intercept)

# Save the updated embeddings and font size mappings
os.makedirs(arrays_path, exist_ok=True)
np.savez_compressed(os.path.join(arrays_path, 'vars.npz'), a=updated_slope, b=updated_intercept)
np.savez_compressed(os.path.join(arrays_path, 'embeddings.npz'), a=updated_faces, b=updated_names)

print("Processing complete. Embeddings and font size mappings have been updated.")
