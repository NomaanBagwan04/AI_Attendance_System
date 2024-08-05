import cv2
import numpy as np
from mtcnn import MTCNN
import torch
from sklearn.preprocessing import LabelEncoder

detector = MTCNN()

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(image_rgb)
    
    if faces:
        x, y, width, height = faces[0]['box']
        face = image_rgb[y:y+height, x:x+width]
        face = cv2.resize(face, (160, 160))
        return face
    return None

def get_embeddings(facenet, image_tensor):
    with torch.no_grad():
        return facenet(image_tensor)

def recognize_face(recognition_model, le, embedding):
    with torch.no_grad():
        output = recognition_model(embedding)
        _, predicted = torch.max(output, 1)
        return le.inverse_transform(predicted.cpu().numpy())[0]
