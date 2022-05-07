from coco_utils import CocoDetection,ConvertCocoBase
import transforms as T
import torch
import utils




def get_transform(train):
    transforms = [ConvertCocoBase()]
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

dataset = CocoDetection(img_folder='../data_bccd_aug_coco/train',
                        ann_file='../data_bccd_aug_coco/train/_annotations.coco.json' ,
                        transforms= get_transform(train=True) )
dataset_test = CocoDetection(img_folder='../data_bccd_aug_coco/valid',
                             ann_file='../data_bccd_aug_coco/valid/_annotations.coco.json' ,
                             transforms= get_transform(train=False) )




def get_bccd_cocostyle_loader():
# define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=4, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)

    return data_loader,data_loader_test

if __name__ == "__main__":
    data_loader,data_loader_test = get_bccd_cocostyle_loader()
    for data in data_loader:
        images, labels = data
        break;