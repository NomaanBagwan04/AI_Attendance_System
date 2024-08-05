import pandas as pd
from facenet_pytorch import InceptionResnetV1, MTCNN
import torch
from PIL import Image
import os
from datetime import datetime

# Initialize models (consider loading them once and reusing)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
recognition_model = torch.load('models/face_recognition_model.pth').to(device)

def load_all_students(class_level):
    # Load all students from a CSV or database
    # Example CSV format: student_id, student_name
    all_students_path = f'data/{class_level}_students.csv'
    if os.path.exists(all_students_path):
        df = pd.read_csv(all_students_path)
        return df['student_id'].tolist(), df['student_name'].tolist()
    else:
        print(f"No student data found for class level {class_level}.")
        return [], []

def process_image(filepath, subject, class_level):
    # Load image
    image = Image.open(filepath).convert('RGB')
    
    # Initialize MTCNN for face detection
    mtcnn = MTCNN(keep_all=True, device=device)
    faces = mtcnn(image)
    
    # Load all students for the class
    all_student_ids, all_student_names = load_all_students(class_level)
    
    # Check if any faces are detected
    if faces is not None and len(faces) > 0:
        # Get face embeddings
        embeddings = facenet(faces).detach().cpu()
        
        # Identify faces using the recognition model
        # Placeholder for actual recognition code
        # Example: embeddings = recognition_model(embeddings)
        # recognized_ids = decode_embeddings(embeddings)
        
        # Assuming you have some method to decode embeddings to IDs
        # For now, use empty list for demonstration
        recognized_ids = []  # Replace with actual ID decoding
        
        # Determine absent students
        absent_ids = set(all_student_ids) - set(recognized_ids)
        present_ids = set(recognized_ids)
        
        # Prepare attendance record
        records_path = 'data/attendance_records/attendance_records.csv'
        date_now = datetime.now().strftime('%Y-%m-%d')
        time_now = datetime.now().strftime('%H:%M:%S')
        
        # Prepare records for present students
        present_data = {
            'student_id': list(present_ids),
            'date': [date_now] * len(present_ids),
            'time': [time_now] * len(present_ids),
            'status': ['Present'] * len(present_ids),
            'subject': [subject] * len(present_ids),
            'class': [class_level] * len(present_ids)
        }
        
        # Prepare records for absent students
        absent_data = {
            'student_id': list(absent_ids),
            'date': [date_now] * len(absent_ids),
            'time': [time_now] * len(absent_ids),
            'status': ['Absent'] * len(absent_ids),
            'subject': [subject] * len(absent_ids),
            'class': [class_level] * len(absent_ids)
        }
        
        # Write attendance data to CSV
        df_present = pd.DataFrame(present_data)
        df_absent = pd.DataFrame(absent_data)
        
        if os.path.exists(records_path):
            df_present.to_csv(records_path, mode='a', header=False, index=False)
            df_absent.to_csv(records_path, mode='a', header=False, index=False)
        else:
            df_present.to_csv(records_path, index=False)
            df_absent.to_csv(records_path, index=False)
    else:
        print("No faces detected in the image.")
        # Mark all students as absent if no faces detected
        absent_data = {
            'student_id': all_student_ids,
            'date': [datetime.now().strftime('%Y-%m-%d')] * len(all_student_ids),
            'time': [datetime.now().strftime('%H:%M:%S')] * len(all_student_ids),
            'status': ['Absent'] * len(all_student_ids),
            'subject': [subject] * len(all_student_ids),
            'class': [class_level] * len(all_student_ids)
        }
        
        df_absent = pd.DataFrame(absent_data)
        if os.path.exists(records_path):
            df_absent.to_csv(records_path, mode='a', header=False, index=False)
        else:
            df_absent.to_csv(records_path, index=False)
