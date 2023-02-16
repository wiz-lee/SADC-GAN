import os

from PIL import Image
from torch.utils import data
from torchvision import transforms


pre_proc = transforms.Compose([transforms.ToTensor()])


class TestDataset(data.Dataset):
    def __init__(self, data_dir):
        super().__init__()
        dirname = os.listdir(data_dir)
        for sub_dir in dirname:
            temp_path = os.path.join(data_dir, sub_dir)
            if sub_dir == 'Inf':
                self.inf_path = temp_path
            elif sub_dir == 'Vis':
                self.vis_path = temp_path

        assert hasattr(self, 'inf_path') and hasattr(self, 'vis_path'), 'check folder name'
        self.name_list = os.listdir(self.inf_path)
        self.name_list_ = os.listdir(self.vis_path)
        assert len(self.name_list) == len(self.name_list_), 'check number of pairs'
        self.transform = pre_proc

    def __getitem__(self, index):
        name = self.name_list[index] 
        assert self.name_list_[index] == name, 'image pair should have the same name'

        inf_image = Image.open(os.path.join(self.inf_path, name)).convert('L') 
        vis_image = Image.open(os.path.join(self.vis_path, name)).convert('L')
        inf_image = self.transform(inf_image)
        vis_image = self.transform(vis_image)

        return inf_image, vis_image, name

    def __len__(self):
        return len(self.name_list)




if __name__ == '__main__':
    ...
