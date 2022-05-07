import numpy as np
import os
from PIL import Image
import torch
import transforms as T
import utils

class PennFudanDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


#TODO 这里测试机随机取样，对测试对比评价是无法追踪的，由这行导致的indices = torch.randperm(len(dataset)).tolist(),
#dataset_test需要固定，而不是随机抽样，本身数据集太少，导致小目标分布有可能造成实验的不确定性
def getloader_PennFudanPed():
    # use our dataset and defined transformations
    root='..'
    dataset = PennFudanDataset(root+'/data/PennFudanPed', get_transform(train=True))
    dataset_test = PennFudanDataset(root+'/data/PennFudanPed', get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()

    train_indices=[48, 56, 133, 76, 71, 44, 29, 58, 149, 115, 12, 78, 151, 168, 155, 128, 43, 113, 82, 19, 169, 94, 96, 95, 30, 42,
     80, 41, 132, 162, 134, 122, 50, 120, 111, 136, 108, 54, 126, 124, 129, 88, 139, 59, 70, 145, 118, 81, 121, 53, 74,
     112, 107, 45, 18, 83, 100, 75, 119, 116, 97, 33, 101, 140, 37, 147, 125, 79, 165, 72, 156, 5, 89, 102, 158, 21, 51,
     138, 148, 69, 163, 123, 141, 24, 90, 135, 22, 164, 105, 152, 3, 39, 60, 109, 2, 159, 93, 150, 1, 52, 143, 137, 117,
     49, 11, 106, 23, 84, 142, 10, 98, 64, 27, 65, 68, 14, 38, 62, 91, 28]
    dataset = torch.utils.data.Subset(dataset, train_indices)

    test_indices = [40, 0, 8, 110, 6, 4, 7, 130, 161, 15, 61, 154, 167, 17, 85, 86, 87, 153, 31, 166, 127, 9, 114, 92, 144, 146, 26, 32, 99, 73, 55, 131, 16, 66, 20, 157, 77, 35, 13, 160, 36, 103, 67, 63, 47, 57, 46, 34, 104, 25] #固定测试集
    print('train_indices')
    print(train_indices)
    print('test_indices')
    print(test_indices)
    dataset_test = torch.utils.data.Subset(dataset_test, test_indices)

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)

    return data_loader,data_loader_test

if __name__ == "__main__":
    data_loader,data_loader_test = getloader_PennFudanPed()
    for data in data_loader:
        images, labels = data
        break;