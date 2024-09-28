import os
import numpy as np
from PIL import Image, ImageFont
from facenet_pytorch import MTCNN, InceptionResnetV1
from detect_faces import detect_faces
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# Paths
faces_path = './data/output_crop_faces/'
images_path = './data/input_faces/'
font_path = './data/font/Calibri Regular.ttf'
arrays_path = './data/arrays/'

# Load pre-trained MTCNN and Inception Resnet V1 models
device = 'cuda' if tf.config.list_physical_devices('GPU') else 'cpu'
mtcnn = MTCNN(keep_all=True, device=device)
inception = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def extract_face_embeddings(image):
    """Extract embeddings for faces in an image."""
    try:
        faces = mtcnn(image)
        if faces is None:
            return np.array([])
        embeddings = inception(faces.to(device)).detach().cpu().numpy()
        return embeddings
    except Exception as e:
        print(f"Error processing image: {e}")
        return np.array([])

def load_existing_data():
    """Load existing embeddings and font size mappings."""
    try:
        with np.load(os.path.join(arrays_path, 'embeddings.npz')) as data:
            existing_faces = data['a']
            existing_names = data['b']

        with np.load(os.path.join(arrays_path, 'vars.npz')) as data:
            existing_slope = data['a']
            existing_intercept = data['b']

        return existing_faces, existing_names, existing_slope, existing_intercept
    except FileNotFoundError:
        print("No existing data found. Starting fresh.")
        return np.array([]), np.array([]), np.array([]), np.array([])

def update_embeddings(new_faces, new_names, existing_faces, existing_names):
    """Update and de-duplicate embeddings."""
    if len(existing_faces) == 0:
        return new_faces, new_names

    all_faces = np.vstack((existing_faces, new_faces))
    all_names = np.concatenate((existing_names, new_names))

    unique_names, indices = np.unique(all_names, return_index=True)
    unique_faces = all_faces[indices]

    return unique_faces, unique_names

def update_font_size_mappings(new_names, existing_names, existing_slope, existing_intercept):
    """Calculate or update font size mappings."""
    combined_names = np.concatenate((existing_names, new_names))
    slopes = np.zeros(len(combined_names))
    intercepts = np.zeros(len(combined_names))

    for name in combined_names:
        try:
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
        except Exception as e:
            print(f"Error calculating font size mapping for {name}: {e}")
    
    return slopes, intercepts

def data_augmentation(image):
    """Apply data augmentation techniques."""
    augmented_images = []
    try:
        augmented_images.append(image)
        augmented_images.append(image.rotate(15))
        augmented_images.append(image.rotate(-15))
        augmented_images.append(image.transpose(Image.FLIP_LEFT_RIGHT))
    except Exception as e:
        print(f"Error during data augmentation: {e}")
    return augmented_images

def process_image(img_path):
    """Process an image for face embeddings."""
    try:
        img = Image.open(img_path)
        augmented_images = data_augmentation(img)  # Apply augmentation
        faces = []
        names = []
        for augmented_img in augmented_images:
            embeddings = extract_face_embeddings(augmented_img)
            if embeddings.size > 0:
                faces.append(embeddings)
                names.extend([os.path.basename(os.path.dirname(img_path))] * embeddings.shape[0])
        return np.vstack(faces) if faces else np.array([]), np.array(names)
    except Exception as e:
        print(f"Error processing file {img_path}: {e}")
        return np.array([]), np.array([])

# User choice for face detection
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

# Process new faces
new_faces = []
new_names = []

with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
    results = list(executor.map(process_image, [os.path.join(root, file) for root, _, files in os.walk(faces_path) for file in files if file.endswith(('.jpg', '.jpeg', '.png'))]))

for faces, names in results:
    if faces.size > 0:
        new_faces.append(faces)
        new_names.extend(names)

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
