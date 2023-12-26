from site import USER_BASE
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from PIL import Image
from typing import Tuple
from datasets.utils.continual_dataset import ContinualDataset
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
import os
from backbone.EfficientNet import mammoth_efficientnet
from datasets.transforms.denormalization import DeNormalize


def store_ravdess_dataset(domain_id, transform,  setting):
    train_dataset  = RavdessDataset(domain_id, verbose=True) 
    test_dataset  = RavdessDataset(domain_id=domain_id, type='val', verbose=True) 
    train_loader = DataLoader(train_dataset,
                              batch_size=setting.args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, 
                             batch_size=setting.args.batch_size, shuffle=False, drop_last=True)
    setting.test_loaders.append(test_loader)
    setting.train_loader = train_loader
    return train_loader, test_loader


def store_test_ravdess_dataset(domain_id, setting):
    test_dataset  = RavdessDataset(domain_id=domain_id, type='val', verbose=True)     
    test_loader = DataLoader(test_dataset,
                             batch_size=setting.args.batch_size, shuffle=False)
    setting.test_loader = test_loader
    return test_loader

class RavdessDataset(Dataset):
    """
    """
    def __init__(self, domain_id, name='Ravdess', img_size = 112, dir='.data/RAVDESS', 
                 type='train', verbose=False, normalize=False, transform=False):
        """
        Args:
        name: Name of the image dataset
        output image size: 32
        dir: root directory to the dataset
        type: train [Default]
        """
        self.dataset_name = name
        self.img_size = img_size
        self.root_dirs = os.path.join(dir, str(type),str(domain_id))
        
        self.type = type
        self.__process__()

        if verbose:
            print("{}: Domain {}: '{}'-dataset consisting of {} samples of size {}".format(name, domain_id, type, self.__len__(), self.img_size))
         
    def __getitem__(self, idx:int) -> Tuple[type(Image), int, type(Image)]:
        
        MEAN, STD = (0.4856, 0.4994, 0.4324), (0.2272, 0.2226, 0.2613)
        IMG_SIZE = self.img_size
        RESIZE_IMG = 128
        if self.type=='train':
            TRANSFORM = transforms.Compose(
                [
                transforms.Resize((RESIZE_IMG, RESIZE_IMG)),
                transforms.RandomCrop(IMG_SIZE, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(MEAN,
                                    STD )])
            transform = TRANSFORM

        else:

            TEST_TRANSFORM = transforms.Compose(
            [transforms.Resize((IMG_SIZE,IMG_SIZE)), transforms.ToTensor(), transforms.Normalize(
            MEAN,STD)])
            transform = TEST_TRANSFORM
            
            
        not_aug_transform = transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)),transforms.ToTensor()])
        
        sample = self.dataset[idx][0]
        target = self.dataset[idx][1]
        original_img = sample.copy()
        
        sample = transform(sample)

        not_aug_img = not_aug_transform(original_img)

        return sample, target, not_aug_img
    
    def __len__(self):
        """ Get the dataset number of samples
        Args:
        Returns:
            [integer]: [returns the total size of the tensor]
        """
        self.length = self.dataset.__len__()
        return self.length
    
    def __process__(self):
       
        if self.img_size <= 224 and self.img_size >= 32:
            self.dataset = self.generate_tensor(img_size=self.img_size)
        else:
            print("{} dataset does not matches with dataset requirement")
                    
    def generate_tensor(self, img_size=244, normalize=False):
        """[summary]

        Args:
            img_size (int, optional): [description]. Defaults to 244.
            normalize (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """
        return (ImageFolder(self.root_dirs))
                   
    def class_idx(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return self.dataset.class_to_idx        
    
    def __repr__(self):
        inform = f"""<{self.dataset_name} {self.length} {self.type} {self.root_dirs} {self.transform}>"""
        return inform
    
    def __str__(self):
        self.__len__()
        inform = f"""Dataset {self.dataset_name}
                    \tNumber of datapoints: {self.length}
                    \tSplit: {self.type}
                    \tRoot Location: {self.root_dirs}
                    \tTransforms (if any):
                    \t{self.transform}"""
        return inform
        


class RAVDESS(ContinualDataset):
    NAME = 'ravdess'
    SETTING = 'domain-il'
    N_CLASSES_PER_TASK = 7
    N_TASKS = 3 #Number of Domains
    DOMAIN_ID ='1'
    MEAN, STD = (0.4856, 0.4994, 0.4324), (0.2272, 0.2226, 0.2613)
    IMG_SIZE = 112
    IMG_RESIZE = 128
    TRANSFORM = transforms.Compose(
            [
            transforms.Resize(IMG_SIZE),
            transforms.RandomCrop(IMG_SIZE, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(MEAN,
                                  STD )])
    

    def get_nonperm_data_loaders(self):
        transform = transforms.Compose((transforms.ToTensor(), ))      
        train, test = store_ravdess_dataset(1, transform,  self)        
        return train, test


    def data_loader_with_did(self, did):
        domain_id = self.get_did(domain_id = did)
        transform = transforms.Compose((transforms.ToTensor(), ))
        train, test = store_ravdess_dataset(domain_id, transform,  self)    
        return train, test
    
    def test_data_loader_with_did(self, did):
        domain_id = self.get_did(domain_id = did)
        transform = transforms.Compose((transforms.ToTensor(), ))
        test = store_test_ravdess_dataset(domain_id,  self)    
        return test

    def get_did(self, domain_id):
        DOMAIN_ID = domain_id
        return DOMAIN_ID

    @staticmethod
    def get_backbone(args=None):
        return mammoth_efficientnet(RAVDESS.N_CLASSES_PER_TASK, model_name="efficientnet-b0",pretrained=True)
        # return resnet50(Ravdess.N_CLASSES_PER_TASK)

    
    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), RAVDESS.TRANSFORM])
        return transform

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_scheduler(model, args):
        return None

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(RAVDESS.MEAN,
                                        RAVDESS.STD)
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(RAVDESS.MEAN,
                                RAVDESS.STD)
        return transform


def main():
   r = RAVDESS(verbose=True) 

if __name__ == '__main__':
    main()