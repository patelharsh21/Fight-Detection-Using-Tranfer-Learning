import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import cv2
import numpy as np


fight_path_template = 'dataset/fight/fi{num:03d}.mp4'
nofight_path_template = 'dataset/noFight/nofi{num:03d}.mp4'

class DataLoading() :
    def load_video_with_resizing_and_frame_handling(self,video_path, target_frames=32, target_size=(112, 112)):
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
        return video_tensor # numpy array
     
    def get_data(self) :
        num_samples=150
        data=[]
        labels=[]
        count=0
        for i in range(1, num_samples + 1):
                    video_path = fight_path_template.format(num=i)
                    video_tensor = self.load_video_with_resizing_and_frame_handling(video_path)
                    if video_tensor is not None:
                        count+=1
                        data.append(video_tensor)
                        labels.append(1)


        for i in range(1, num_samples + 1):
                    video_path = nofight_path_template.format(num=i)
                    video_tensor = self.load_video_with_resizing_and_frame_handling(video_path)
                    if video_tensor is not None:
                        count+=1
                        data.append(video_tensor)
                        labels.append(0)
        return data,labels

instance=DataLoading()
data,labels=instance.get_data()
# Assuming tensor_X and tensor_Y are already defined
# tensor_X: [300, 32, 112, 112, 3]
# tensor_Y: [300]
data=np.array(data)
labels=np.array(labels)
tensor_X=torch.tensor(data)
tensor_Y=torch.tensor(labels)


# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    tensor_X, tensor_Y, test_size=0.2, random_state=42, stratify=tensor_Y
)

# Permute the tensors to [batch_size, channels, depth, height, width]
X_train = X_train.permute(0, 4, 1, 2, 3).float()  # Shape: [240, 3, 32, 112, 112]
X_test = X_test.permute(0, 4, 1, 2, 3).float()    # Shape: [60, 3, 32, 112, 112]

# Ensure labels are float for BCEWithLogitsLoss
y_train = y_train.float()  # Shape: [240]
y_test = y_test.float()    # Shape: [60]

class VideoDataset(Dataset):
    def __init__(self, videos, labels):
        self.videos = videos  # Tensor containing video data [batch, 3, 32, 112, 112]
        self.labels = labels  # Tensor containing labels [batch]

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video = self.videos[idx]    # [3, 32, 112, 112]
        label = self.labels[idx]    # Scalar float (0.0 or 1.0)
        return video, label

# Create Dataset instances
dataset_train = VideoDataset(X_train, y_train)
dataset_test = VideoDataset(X_test, y_test)

# Create DataLoader instances
batch_size = 2
data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
data_loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4)


import torchvision
# Load the pretrained r3d_18 model
pretrained_model = torchvision.models.video.r3d_18(pretrained=True)


class Simple3DCNN(nn.Module):
    def __init__(self):
        super(Simple3DCNN, self).__init__()
        # Correct in_channels to 3 for RGB
        # self.conv1 = nn.Conv3d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        # self.pool = nn.MaxPool3d(kernel_size=2, stride=2)  # Halves the spatial dimensions
        # self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # # Adaptive pooling to reduce tensor to (batch_size, 32, 1, 1, 1)
        # self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc1 = nn.Linear(400, 128)  # Adjusted input features
        self.fc2 = nn.Linear(128, 1)   # Output layer for binary classification

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))       # [batch, 16, 16, 56, 56]
        x = self.pool(F.relu(self.conv2(x)))       # [batch, 32, 8, 28, 28]
        x = self.adaptive_pool(x)                   # [batch, 32, 1, 1, 1]
        x = x.view(x.size(0), -1)                   # [batch, 32]
        x = F.relu(self.fc1(x))                     # [batch, 128]
        x = self.fc2(x)                             # [batch, 1]
        return x

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Initialize the model and move to device
model = Simple3DCNN().to(device)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Initialize lists to store loss and accuracy
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

# Training and Testing Loop
num_epochs = 20

for epoch in tqdm(range(num_epochs), desc='Training Epochs'):
    # Training Phase
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_videos, batch_labels in data_loader_train:
        # Move data to device
        batch_videos = batch_videos.to(device)    # [batch_size, 3, 32, 112, 112]
        batch_labels = batch_labels.to(device)    # [batch_size]

        optimizer.zero_grad()  # Zero the gradients

        outputs = model(batch_videos)            # [batch_size, 1]
        outputs = outputs.squeeze(1)             # [batch_size]
        loss = criterion(outputs, batch_labels)  # BCEWithLogitsLoss

        loss.backward()          # Backward pass
        optimizer.step()         # Update weights

        running_loss += loss.item() * batch_videos.size(0)

        # Calculate accuracy
        preds = torch.sigmoid(outputs) >= 0.5  # Apply sigmoid and threshold at 0.5
        correct += (preds.float() == batch_labels).sum().item()
        total += batch_labels.size(0)

    epoch_loss = running_loss / len(data_loader_train.dataset)
    epoch_acc = 100 * correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    # Testing Phase
    model.eval()
    test_running_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for batch_videos, batch_labels in data_loader_test:
            batch_videos = batch_videos.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_videos)
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, batch_labels)

            test_running_loss += loss.item() * batch_videos.size(0)

            # Calculate accuracy
            preds = torch.sigmoid(outputs) >= 0.5
            test_correct += (preds.float() == batch_labels).sum().item()
            test_total += batch_labels.size(0)

    test_epoch_loss = test_running_loss / len(data_loader_test.dataset)
    test_epoch_acc = 100 * test_correct / test_total
    test_losses.append(test_epoch_loss)
    test_accuracies.append(test_epoch_acc)

    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, '
          f'Test Loss: {test_epoch_loss:.4f}, Test Acc: {test_epoch_acc:.2f}%')

# Visualize the training and testing loss and accuracy
plt.figure(figsize=(14, 6))

# Loss Plot
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss', marker='o')
plt.plot(range(1, num_epochs+1), test_losses, label='Test Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss over Epochs')
plt.legend()
plt.grid(True)

# Accuracy Plot
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs+1), train_accuracies, label='Train Accuracy', marker='o')
plt.plot(range(1, num_epochs+1), test_accuracies, label='Test Accuracy', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Test Accuracy over Epochs')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
