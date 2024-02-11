import os

from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
from torchvision.io import read_image
import torchvision.transforms as transforms


class Music_Dataset(Dataset):
    def __init__(self, annotations_file, img_dir="./mel_spec"):
        self.padding = "max_length"
        self.text_column_name = 'text'
        self.dateset=pd.read_csv(annotations_file)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.dateset)

    def __getitem__(self, idx):
        text = self.dateset.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, self.dateset.iloc[idx, 1])

        image = Image.open(img_path)
        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts the image to a float tensor in the range [0.0, 1.0]
        ])

        # Apply the transform to the image
        image = transform(image)

        label = self.dateset.iloc[idx, 2]
        sample={"text":text, 'image':image}
        return sample, label

