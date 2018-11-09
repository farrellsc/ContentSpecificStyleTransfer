from torch.utils.data import Dataset
import os
from skimage import io, transform
import logging


class PlacesDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.img_names = os.listdir(img_dir)

    def __len__(self):
        len(self.img_names)

    def __getitem__(self, idx):
        img_full_name = os.path.join(self.img_dir, self.img_names[idx])
        image = io.imread(img_full_name)
        return image


if __name__ == "__main__":
    data = PlacesDataset("/Users/rl/PycharmProjects/SuperStyleTransfer/data/testSetPlaces205_resize/testSet_resize")
    print(data[0].shape)
