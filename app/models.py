import torch
from facenet_pytorch import InceptionResnetV1
import torch.nn as nn

class FaceRecognitionModel(nn.Module):
    def __init__(self, embedding_size, num_classes):
        super(FaceRecognitionModel, self).__init__()
        self.fc = nn.Linear(embedding_size, num_classes)
    
    def forward(self, x):
        return self.fc(x)

def load_models():
    facenet = InceptionResnetV1(pretrained='vggface2').eval()
    recognition_model = FaceRecognitionModel(embedding_size=512, num_classes=100)  # Adjust num_classes
    facenet.load_state_dict(torch.load('models/facenet_model.pth'))
    recognition_model.load_state_dict(torch.load('models/face_recognition_model.pth'))
    
    facenet.eval()
    recognition_model.eval()
    
    return facenet, recognition_model
