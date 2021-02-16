import os
import io
import random
from math import log10

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torchvision import transforms as trans

class Dataset(object):
    def __init__(self, images_dir, patch_size=48, jpeg_quality=40, transforms=None, train=False, count=None):
        self.images = os.walk(images_dir).__next__()[2]
        self.images_path = []
        self.mean = 0.0
        self.std = 0.0
        for img_file in self.images:
            if img_file.endswith((".ppm")):
                try:
                    label = Image.open(os.path.join(images_dir, img_file))
                    self.images_path.append(os.path.join(images_dir, img_file))
                    #print(img_file)
                except:
                    print(f"Image {os.path.join(images_dir, img_file)} didn't get loaded")
        
        if count:
            self.images_path = random.sample(self.images_path, count)
        self.patch_size = patch_size
        self.jpeg_quality = jpeg_quality
        self.transforms = transforms
        self.train = train
        
    def __getitem__(self, idx):
        label = Image.open(self.images_path[idx]).convert('RGB')


        # additive jpeg noise
        buffer = io.BytesIO()
        if self.train:
            label.save(buffer, format='jpeg', quality=self.jpeg_quality)
        else:
            label.save(buffer, format='jpeg', quality=self.jpeg_quality)

        input = Image.open(buffer).convert('RGB')

        if self.transforms is not None:
            input = self.transforms(input)
            label = self.transforms(label)
        #print("Image transformed")
        return input, label

    def __len__(self):
        return len(self.images_path)
