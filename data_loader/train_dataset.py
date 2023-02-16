from torch.utils import data
from torchvision import transforms
import h5py
import numpy as np


pre_proc = transforms.Compose([transforms.ToTensor()])

class TrainDataset(data.Dataset):
    def __init__(self, h5file):
        super().__init__()
        
        self.transform = pre_proc
        f = h5py.File(h5file, 'r')
        sources = f['data'][:]
        sources = np.transpose(sources, (0, 3, 2, 1))
        sources = np.uint8(sources * 255)

        self.vis_images = sources[:, :, :, 0:1]
        self.inf_images = sources[:, :, :, 1:2]             

    def __getitem__(self, index):
        inf_image = self.inf_images[index]
        vis_image = self.vis_images[index]
        inf_image = self.transform(inf_image)
        vis_image = self.transform(vis_image)

        return inf_image, vis_image

    def __len__(self):
        return self.inf_images.shape[0]




if __name__ == '__main__':
    ...
