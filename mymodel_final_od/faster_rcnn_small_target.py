#!/usr/bin/python

import torch
from engine import train_one_epoch, evaluate
from PennFudanPed_dataloader import getloader_PennFudanPed
from rcnn_new_loss_model import get_model
from voc_cocostyle_dataloader import get_voc_cocostyle_loader
from bccd_cocostyle_dataloader import get_bccd_cocostyle_loader

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

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        #evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    print("That's it!")

    PATH = '../my_model_save/frcnn_' + 'newloss_' +dataset_name+ '_net.pth'
    if is_origin == False:
        PATH = '../my_model_save/frcnn_'+'orign_'+dataset_name+'_net.pth'

    #model.load_state_dict(torch.load(PATH))
    torch.save(model.state_dict(), PATH)

if __name__ == "__main__":
    main()
