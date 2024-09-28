# Facial Recognition Attendance System

This project implements a **Facial Recognition System** that automatically tracks attendance by detecting faces from images uploaded to Firebase. It uses deep learning models for face detection and recognition, allowing for continuous learning by adding new faces and updating embeddings.

## Features

- **Face Detection and Recognition** with MTCNN and InceptionResnetV1 from `facenet-pytorch`
- **Firebase Integration** for downloading images
- **MySQL Database** for attendance tracking (check-in and check-out)
- **Data Augmentation** for improved recognition accuracy
- **Multi-threaded Processing** for faster embedding extraction
- **Automated Scheduling** for daily processing
- **CSV Output**: Generates daily recognition results in CSV format

## Project Structure

```plaintext
├── Trainer/                             # Training scripts and data
│   ├── data/
│   │    └── arrays/                     # Pre-trained embeddings and mappings
│   │    └── font/                       # Font files for rendering names
│   │    └── input_faces/                # Input face images for training
│   │    └── output_crop_faces/          # Cropped face images after detection
│   ├── script/
│   │    └── detect_faces.py             # Face detection script
│   │    └── generate_data_augmented.py  # Training script with data augmentation
│   │
│   ├── image_recognize.py
│   ├── Video_recognize.py
│
├── Recognizer/                         # Face recognition processes
│   └── arrays/                         # Stores embeddings and recognition data
│   └── output/                         # Detected face images from recognition
│   └── main.py                         # Main script for training and recognition
│   └── scheduler.py                    # Script for automated scheduling
│   └── crediential.json                # Firebase credentials
│
│   requirements.txt                    # Python dependencies
│   README.md                           # Project documentation
     
```

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/afzalabdullah/facial-recognition-attendance.git
   cd facial-recognition-attendance
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Firebase Credentials**:

   Download the Firebase service account key (JSON) and save it as `crediential.json` in the Recognizer directory.

4. **Set Up MySQL Database**:

   ```sql
   CREATE DATABASE ams;
   CREATE TABLE attendance (
       id INT AUTO_INCREMENT PRIMARY KEY,
       name VARCHAR(255),
       date DATE,
       checkin_time TIME,
       checkout_time TIME
   );
   ```

   Update the MySQL credentials in the `main.py` script.

## Usage

### Training (Embedding Extraction)

1. **Organize Training Images**:
   - Store images in `./data/input_faces/` with subdirectories for each person.

   ```plaintext
   ./data/input_faces/
   ├── person1/
   ├── person2/
   ```

2. **Run Training Script**:

   ```bash
   python generate_data_augmented.py
   ```

3. **Select Option for Face Detection**:

   Choose between having existing faces or detecting new ones.

4. **Results**:
   - Embeddings are saved in `./arrays/embeddings.npz`.
   - Font size mappings in `./arrays/vars.npz`.

### Attendance Tracking (Recognition)

1. **Prepare New Images**:
   Upload images to Firebase Storage in the specified format.

2. **Run Recognition Script**:

   ```bash
   python main.py
   ```

### Automated Scheduling

Set up a daily schedule using `cron` (Linux) or Task Scheduler (Windows) to run the script automatically.

## Output

- **CSV Files**: Daily recognition results saved as `recognition_results_YYYYMMDD.csv`.
- **Detected Faces**: Cropped images stored in the `output/` folder.

## Data Augmentation

The system applies augmentation techniques (rotation, flipping) to improve recognition accuracy.

## Multi-threading

Utilizes multi-threading for efficient processing during embedding extraction.

## Contributing

Feel free to open issues or submit pull requests to enhance the project.

## License

This project is licensed under the MIT License.

## Requirements

- `facenet-pytorch`
- `tensorflow`
- `Pillow`
- `numpy`
- `scikit-learn`
- `mysql-connector-python`
- `firebase-admin`

All dependencies can be installed via `requirements.txt`.

## Acknowledgments

This project leverages pre-trained models from `facenet-pytorch`. Thanks to the contributors of these models.
