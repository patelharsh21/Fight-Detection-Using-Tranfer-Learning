import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader, Dataset

# Define paths
fight_path_template = 'dataset/fight/fi{num:03d}.mp4'
nofight_path_template = 'dataset/noFight/nofi{num:03d}.mp4'

# Load video function (same as before but modularized)
def load_video_with_resizing_and_frame_handling(video_path, target_frames=64, target_size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, target_size)
        normalized_frame = resized_frame / 255.0
        frames.append(normalized_frame)

    cap.release()
    video_tensor = np.array(frames)

    num_frames = video_tensor.shape[0]
    if num_frames > target_frames:
        indices = np.linspace(0, num_frames - 1, target_frames).astype(int)
        video_tensor = video_tensor[indices]
    elif num_frames < target_frames:
        pad_length = target_frames - num_frames
        padding = np.zeros((pad_length, target_size[0], target_size[1], 3))
        video_tensor = np.concatenate((video_tensor, padding), axis=0)

    return video_tensor

# Dataset class for fight/noFight videos
class FightDataset(Dataset):
    def __init__(self, num_samples=150, target_frames=64, target_size=(224, 224)):
        self.num_samples = num_samples
        self.target_frames = target_frames
        self.target_size = target_size
        self.data = []
        self.labels = []

        # Load fight videos
        for i in range(1, self.num_samples + 1):
            video_path = fight_path_template.format(num=i)
            video_tensor = load_video_with_resizing_and_frame_handling(video_path, target_frames, target_size)
            if video_tensor is not None:
                self.data.append(video_tensor)
                self.labels.append(1)  # Label 1 for fight

        # Load noFight videos
        for i in range(1, self.num_samples + 1):
            video_path = nofight_path_template.format(num=i)
            video_tensor = load_video_with_resizing_and_frame_handling(video_path, target_frames, target_size)
            if video_tensor is not None:
                self.data.append(video_tensor)
                self.labels.append(0)  # Label 0 for no fight

        # Convert lists to numpy arrays
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video = self.data[idx]
        label = self.labels[idx]
        return torch.FloatTensor(video).permute(3, 0, 1, 2), torch.LongTensor([label])

# Instantiate dataset and dataloader
dataset = FightDataset(num_samples=150)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# I3D Model for Transfer Learning
class I3DTransferModel(nn.Module):
    def __init__(self, num_classes=2):
        super(I3DTransferModel, self).__init__()
        self.i3d = models.video.r3d_18(pretrained=True)  # Using pretrained 3D ResNet as a substitute for I3D
        self.i3d.fc = nn.Linear(self.i3d.fc.in_features, num_classes)  # Replace the final layer

    def forward(self, x):
        return self.i3d(x)

# Instantiate the model
model = I3DTransferModel(num_classes=2)

# Training Setup
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

# Train the model
train(model, dataloader, criterion, optimizer, num_epochs=10)
