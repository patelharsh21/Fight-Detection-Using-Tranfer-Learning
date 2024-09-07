import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader, Dataset

# Define paths
fight_path_template = 'dataset/fight/fi{num:03d}.mp4'
nofight_path_template = 'dataset/noFight/nofi{num:03d}.mp4'

def load_video_with_resizing_and_frame_handling(video_path, target_frames=32, target_size=(112, 112)):
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

class FightDataset(Dataset):
    def __init__(self, num_samples=150, target_frames=32, target_size=(112, 112)):
        self.num_samples = num_samples
        self.target_frames = target_frames
        self.target_size = target_size
        self.data = []
        self.labels = []

        for i in range(1, self.num_samples + 1):
            video_path = fight_path_template.format(num=i)
            video_tensor = load_video_with_resizing_and_frame_handling(video_path, target_frames, target_size)
            if video_tensor is not None:
                self.data.append(video_tensor)
                self.labels.append(1)

        for i in range(1, self.num_samples + 1):
            video_path = nofight_path_template.format(num=i)
            video_tensor = load_video_with_resizing_and_frame_handling(video_path, target_frames, target_size)
            if video_tensor is not None:
                self.data.append(video_tensor)
                self.labels.append(0)

        self.data = np.array(self.data)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video = self.data[idx]
        label = self.labels[idx]
        return torch.FloatTensor(video).permute(3, 0, 1, 2), torch.LongTensor([label])

dataset = FightDataset(num_samples=150)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

class I3DTransferModel(nn.Module):
    def __init__(self, num_classes=2):
        super(I3DTransferModel, self).__init__()
        self.i3d = models.video.r3d_18(pretrained=True)
        self.i3d.fc = nn.Linear(self.i3d.fc.in_features, num_classes)

    def forward(self, x):
        return self.i3d(x)

model = I3DTransferModel(num_classes=2)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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

train(model, dataloader, criterion, optimizer, num_epochs=10)
