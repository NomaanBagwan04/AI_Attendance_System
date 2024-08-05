from flask import Blueprint, request, render_template, jsonify
import torch
from .models import load_models
from .utils import preprocess_image, get_embeddings, recognize_face
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

main = Blueprint('main', __name__)

facenet, recognition_model = load_models()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
facenet.to(device)
recognition_model.to(device)

# Load LabelEncoder
le = LabelEncoder()
le.classes_ = np.load('models/classes.npy')

@main.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        image_path = os.path.join('uploads', file.filename)
        file.save(image_path)

        face = preprocess_image(image_path)
        if face is not None:
            face_tensor = torch.tensor(face, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
            embedding = get_embeddings(facenet, face_tensor)
            student_name = recognize_face(recognition_model, le, embedding)
            return jsonify({'student_name': student_name})
        
        return jsonify({'error': 'No face detected'})

    return render_template('index.html')
