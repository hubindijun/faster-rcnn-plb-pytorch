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

dataset = CocoDetection(img_folder='../data_voc/VOCdevkit/VOC2007/JPEGImages',
                        ann_file='../data_voc/VOCdevkit/VOC2007/ImageSets/Main/coco_train.json' ,
                        transforms= get_transform(train=True) )
dataset_test = CocoDetection(img_folder='../data_voc/VOCdevkit/VOC2007/JPEGImages',
                             ann_file='../data_voc/VOCdevkit/VOC2007/ImageSets/Main/coco_val.json' ,
                             transforms= get_transform(train=False) )




def get_voc_cocostyle_loader():
# define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=4, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)

    return data_loader,data_loader_test

if __name__ == "__main__":
    data_loader,data_loader_test = get_voc_cocostyle_loader()
    count_small = 0
    count_medium = 0
    count_large = 0
    for data in data_loader:
        images, labels = data
        for label in labels:
            boxes = label["boxes"]

            input_width = boxes[:, 2] - boxes[:, 0]
            input_height = boxes[:, 3] - boxes[:, 1]
            areas = input_width * input_height
            areas_list = areas.numpy().tolist()
            for a in areas_list:
                if a<(32*32):
                    count_small= count_small + 1
                elif a<(96*96):
                    count_medium=count_medium + 1
                else:
                    count_large = count_large + 1

    print(count_small)
    print(count_medium)
    print(count_large)

    count_small_test = 0
    count_medium_test = 0
    count_large_test = 0
    for data in data_loader_test:
        images, labels = data
        for label in labels:
            boxes = label["boxes"]

            input_width = boxes[:, 2] - boxes[:, 0]
            input_height = boxes[:, 3] - boxes[:, 1]
            areas = input_width * input_height
            areas_list = areas.numpy().tolist()
            for a in areas_list:
                if a < (32 * 32):
                    count_small_test = count_small_test + 1
                elif a < (96 * 96):
                    count_medium_test = count_medium_test + 1
                else:
                    count_large_test = count_large_test + 1
    print("count_for_test")
    print(count_small_test)
    print(count_medium_test)
    print(count_large_test)