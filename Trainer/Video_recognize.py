import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from PIL import Image, ImageFont, ImageDraw
from sklearn.metrics.pairwise import cosine_similarity
from facenet_pytorch import InceptionResnetV1, MTCNN
import cv2

# Disable TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Paths
font_path = './data/font/Calibri Regular.ttf'
arrays_path = './arrays/'

# Load saved embeddings and font size mappings
embeddings_data = np.load(os.path.join(arrays_path, 'embeddings.npz'))
slope_intercept_data = np.load(os.path.join(arrays_path, 'vars.npz'))

stored_embeddings = embeddings_data['a']
names = embeddings_data['b']
slope = slope_intercept_data['a']
intercept = slope_intercept_data['b']

# Load pre-trained MTCNN and Inception Resnet V1 models
device = 'cuda' if tf.config.list_physical_devices('GPU') else 'cpu'
mtcnn = MTCNN(keep_all=True, device=device)
inception = InceptionResnetV1(pretrained='vggface2').eval().to(device)

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

def recognize_face(frame, threshold=0.7):
    embeddings = extract_face_embeddings(frame)
    if embeddings.size == 0:
        return "No face detected", None

    # Compare with stored embeddings
    similarities = cosine_similarity(embeddings, stored_embeddings)
    best_match_index = np.argmax(similarities.mean(axis=0))
    best_score = similarities.mean(axis=0)[best_match_index]

    if best_score >= threshold:
        best_name = names[best_match_index]
    else:
        best_name = "Unidentified"

    return best_name, best_score

def estimate_font_size(name, w_face):
    index = np.where(names == name)[0][0]
    slope_value = slope[index]
    intercept_value = intercept[index]

    font_size = int(slope_value * w_face + intercept_value)
    return font_size

def render_text(frame, text, x, y, font_size):
    try:
        font = ImageFont.truetype(font_path, font_size)
        pil_image = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil_image)
        draw.text((x, y), text, font=font, fill=(255, 0, 0))
        return np.array(pil_image)
    except Exception as e:
        print(f"Error in render_text: {e}")
        return frame

# Initialize the camera
camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    if not ret:
        break

    # Convert frame to RGB (for PIL)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    
    try:
        # Detect faces
        faces, _ = mtcnn.detect(pil_image)
    except Exception as e:
        print(f"Error in face detection: {e}")
        faces = None

    if faces is not None:
        for box in faces:
            try:
                x_face, y_face, w_face, h_face = int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])
                
                # Recognize face
                recognized_name, confidence = recognize_face(pil_image.crop((x_face, y_face, x_face + w_face, y_face + h_face)))

                if recognized_name != "No face detected" and recognized_name != "Unidentified":
                    # Estimate font size based on face width
                    font_size = estimate_font_size(recognized_name, w_face)
                    
                    # Draw rectangle around face
                    cv2.rectangle(frame, (x_face, y_face), (x_face + w_face, y_face + h_face), (0, 255, 0), 2)
                    
                    # Render the name with the estimated font size
                    frame = render_text(frame, recognized_name, x_face, y_face - 30, font_size)
                else:
                    cv2.rectangle(frame, (x_face, y_face), (x_face + w_face, y_face + h_face), (0, 0, 255), 2)
                    frame = render_text(frame, "Unidentified", x_face, y_face - 30, 20)
            except Exception as e:
                print(f"Error processing face: {e}")
                continue

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
