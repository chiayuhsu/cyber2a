import json
import os
from torch.utils.data import Dataset
from PIL import Image


class RGBDataset(Dataset):
    def __init__(self, split, transform=None):
        """
        Args:
            split (str): One of 'train' or 'valtest' to specify the dataset split.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # Define the directory where images are stored
        self.img_dir = "cyber2a/rts/rgb/"
        
        # Load the list of images based on the split (train/valtest)
        with open("cyber2a/rts/data_split.json") as f:
            data_split = json.load(f)
            
        if split == 'train':
            self.img_list = data_split['train']
        elif split == 'valtest':
            self.img_list = data_split['valtest']
        else:
            raise ValueError("Invalid split: choose either 'train' or 'valtest'")
    
        # Load the image labels
        with open("cyber2a/rts/rts_cls.json") as f:
            self.img_labels = json.load(f)

        # Store the transform to be applied to images
        self.transform = transform

    def __len__(self):
        """Return the total number of images."""
        return len(self.img_list)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the image to retrieve.
        
        Returns:
            tuple: (image, label) where image is the image tensor and label is the corresponding label.
        """
        # Retrieve the image name using the index
        img_name = self.img_list[idx]
      
        # Construct the full path to the image file
        img_path = os.path.join(self.img_dir, img_name)
        
        # Open the image and convert it to RGB format
        image = Image.open(img_path).convert('RGB')
        
        # Get the corresponding label and adjust it to be 0-indexed
        label = self.img_labels[img_name] - 1

        # apply transform if specified
        if self.transform:
            image = self.transform(image)

        return image, label


class MultiModalDataset(Dataset):
    def __init__(self, split, transform=None):
        """
        Args:
            split (str): One of 'train' or 'valtest' to specify the dataset split.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # Define the directories where images are stored
        self.rgb_dir = "cyber2a/rts/rgb/"
        self.ndvi_dir = "cyber2a/rts/ndvi/"
        self.nir_dir = "cyber2a/rts/nir/"

        # Load the list of images based on the split (train/valtest)
        with open("cyber2a/rts/data_split.json") as f:
            data_split = json.load(f)

        if split == "train":
            self.img_list = data_split["train"]
        elif split == "valtest":
            self.img_list = data_split["val_test"]
        else:
            raise ValueError("Invalid split: choose either 'train' or 'valtest'")

        # Load the image labels
        with open("cyber2a/rts/rts_cls.json") as f:
            self.img_labels = json.load(f)

        # Store the transform to be applied to images
        self.transform = transform

    def __len__(self):
        """Return the total number of images."""
        return len(self.img_list)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            dict: A dictionary containing RGB image, depth image, and label.
        """
        # Retrieve the image name using the index
        img_name = self.img_list[idx]

        # Construct the full paths to the RGB and depth image files
        rgb_path = os.path.join(self.rgb_dir, img_name)
        ndvi_path = os.path.join(self.ndvi_dir, img_name)
        nir_path = os.path.join(self.nir_dir, img_name)

        # Open the RGB image and convert it to RGB format
        rgb_image = Image.open(rgb_path).convert("RGB")
        # Open the ndvi image and convert it to grayscale format (L)
        ndvi_image = Image.open(ndvi_path).convert("L")
        # Open the nir image and convert it to grayscale format (L)
        nir_image = Image.open(nir_path).convert("L")

        # Get the corresponding label and adjust it to be 0-indexed
        label = self.img_labels[img_name] - 1

        # Apply transform if specified
        if self.transform:
            rgb_image = self.transform(rgb_image)
            ndvi_image = self.transform(ndvi_image)
            nir_image = self.transform(nir_image)

        # Return a dictionary with all modalities
        return {"rgb": rgb_image, "ndvi": ndvi_image, "nir": nir_image}, label