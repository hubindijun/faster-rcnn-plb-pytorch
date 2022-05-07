#!/usr/bin/python

import torch
from engine import train_one_epoch, evaluate
from PennFudanPed_dataloader import getloader_PennFudanPed
from rcnn_new_loss_model import get_model
from voc_cocostyle_dataloader import get_voc_cocostyle_loader
from bccd_cocostyle_dataloader import get_bccd_cocostyle_loader
import cv2
font = cv2.FONT_HERSHEY_SIMPLEX

bccd_names = {'0': 'background', '1': 'Platelets', '2': 'RBC', '3': 'WBC'}

# import sys
# root='/content/drive/MyDrive'
# #启动colab文件命令 !python3 "/content/drive/MyDrive/my_model/faster_rcnn_small_target.py"
# log_print = open(root+'/logs/old_loss_with_VOC2007.log', 'w')
# sys.stdout = log_print
# sys.stderr = log_print

def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #device = torch.device('cpu')
    #model params
    # 要和数据集对应的上，否则target中出现大于2的值，会报错！ 此处在后续的osr部分，要切记数据的过滤处理，只有训练时才会计算loss
    num_classes = 21
    is_pretrained = True
    is_origin = False #是否采用旧模型，默认否 # 采用小物体损失用最大值除以最小面积
    num_epochs = 10
    dataset_name ='bccd' # or bccd

    # use our dataset and defined transformations

    # define training and validation data loaders
    #data_loader,data_loader_test = getloader_PennFudanPed()
    data_loader, data_loader_test = get_voc_cocostyle_loader()
    if dataset_name =='bccd':
        num_classes = 4
        num_epochs = 15
        data_loader,data_loader_test = get_bccd_cocostyle_loader()

    # get the model using our helper function
    model = get_model(num_classes=num_classes, pretrained=is_pretrained, is_origin=is_origin)
    model_new = model


    PATH_new = '../my_model_save/frcnn_' + 'newloss_' + dataset_name + '_net.pth'
    PATH = '../my_model_save/frcnn_' + 'orign_' + dataset_name + '_net.pth'

    model.load_state_dict(torch.load(PATH,map_location='cpu'))
    model_new.load_state_dict(torch.load(PATH_new,map_location='cpu'))

    # move model to the right device
    #model_new.to(device)
    model.eval()
    model_new.eval()
    # evaluate on the test dataset
    with torch.no_grad():
        for images, targets in data_loader_test:

            images = list(img.to(device) for img in images)
            #outputs = model(images)
            #outputs_new = model_new(images)
            outputs_new =  model(images)
            #对比outputs和outputs2
            print("outputs--info--")
            # for output in outputs_new:
            #
            #     print(output['boxes'].shape)
            #     print(output['labels'].shape)
            #     print(output['scores'].shape)
            # print(outputs_new)
            #
            # print("targets--info--")
            # for target in targets:
            #     print(target['boxes'].shape)
            #     print(target['labels'].shape)
            #    # print(target['scores'].shape)
            #     print(target['image_id'])
            # print(targets)

            #outputs = [{k: v.to("cpu") for k, v in t.items()} for t in outputs]
            # outputs_new = [{k: v.to("cpu") for k, v in t.items()} for t in outputs_new]


            for i in range(len(images)):
                bboxes = []
                ids = []
                scores_val =[]
                img = images[i]
                target_ = outputs_new[i]  # 原始值绘图
                boxes = target_['boxes'].numpy().tolist()
                labels = target_['labels'].numpy().tolist()
                scores = target_['scores'].numpy().tolist()
                for box in boxes:
                    bboxes.append([box[0],
                                   box[1],
                                   # box[0] + box[2],
                                   # box[1] + box[3]
                                   box[2],
                                   box[3]
                                   ])
                for label in labels:
                    ids.append(label)
                for score in scores:
                    scores_val.append(score)

                img = img.permute(1, 2, 0).numpy()
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                for box, id_,score_ in zip(bboxes, ids,scores):
                    x1 = int(box[0])
                    y1 = int(box[1])
                    x2 = int(box[2])
                    y2 = int(box[3])

                    class_name = bccd_names.get(str(id_)) #+":"+str(round(score_,3)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), thickness=1)
                    cv2.putText(img, text=class_name, org=(x1 + 5, y1 + 5), fontFace=font, fontScale=0.5,
                                thickness=1, lineType=cv2.LINE_AA, color=(0, 0, 255))
                cv2.imshow('test', img)
                cv2.waitKey()



    print("That's it!")



if __name__ == "__main__":
    main()
