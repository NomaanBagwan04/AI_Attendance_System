import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

class FaceDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(image_folder))

        for idx, class_name in enumerate(self.classes):
            class_folder = os.path.join(image_folder, class_name)
            for image_name in os.listdir(class_folder):
                self.image_paths.append(os.path.join(class_folder, image_name))
                self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

class FaceRecognitionModel(nn.Module):
    def __init__(self, embedding_size, num_classes):
        super(FaceRecognitionModel, self).__init__()
        self.fc = nn.Linear(embedding_size, num_classes)
    
    def forward(self, x):
        return self.fc(x)

def train_model():
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load dataset
    dataset = FaceDataset(image_folder='data/student_images', transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Load models
    facenet = InceptionResnetV1(pretrained='vggface2').eval()
    recognition_model = FaceRecognitionModel(embedding_size=512, num_classes=len(dataset.classes))

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(recognition_model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        for images, labels in dataloader:
            # Move tensors to GPU if available
            images, labels = images.to('cuda'), labels.to('cuda')
            facenet = facenet.to('cuda')
            recognition_model = recognition_model.to('cuda')
            
            # Extract face embeddings
            embeddings = facenet(images)
            
            # Forward pass
            outputs = recognition_model(embeddings)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    # Save models
    torch.save(facenet.state_dict(), 'models/facenet_model.pth')
    torch.save(recognition_model.state_dict(), 'models/face_recognition_model.pth')

if __name__ == '__main__':
    train_model()
