import cv2
import numpy as np
# import torch
# import torch.nn as nn
# from torchvision import models
# from torch.utils.data import DataLoader, Dataset

# Define paths
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
        print(video_tensor.shape)
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
        return data

instance=DataLoading()
data=instance.get_data()