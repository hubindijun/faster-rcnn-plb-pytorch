
import os

import matplotlib.pyplot as plt
import numpy as np

line1 = " Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = "
line2 = " Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = "
line3 = " Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = "
line4 = " Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = "
line5 = " Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = "
line6 = " Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = "
line7 = " Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = "
line8 = " Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = "
line9 = " Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = "
line10 = " Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = "
line11 = " Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = "
line12 = " Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = "


dfhistory ={
    "origin": [0.126,0.154,0.164,0.190,0.192,0.182,0.185,0.184,0.184,0.183],
     "new":  [0.161,0.150,0.162,0.189,0.194,0.189,0.193,0.192,0.193,0.193],
}


def convert_list2str(arry, name):
    str_value = "\""+name + "\" : ["
    for item in arry:
        str_value += item + ","

    return str_value[::-1].replace(",", "]",1)[::-1]+","


def read_log_file(file_path):
    fp = open(file_path)
    lr = []
    loss = []

    arry1 = []
    arry2 = []
    arry3 = []
    arry4 = []
    arry5 = []
    arry6 = []
    arry7 = []
    arry8 = []
    arry9 = []
    arry10 = []
    arry11 = []
    arry12 = []

    for line in fp.readlines():
        if line.find("Epoch: [15]") != -1:
            break

        if line.find(line1) != -1:
            arry1.append(line[-6:-1])

        if line.find(line2) != -1:
            arry2.append(line[-6:-1])

        if line.find(line3) != -1:
            arry3.append(line[-6:-1])

        if line.find(line4) != -1:
            arry4.append(line[-6:-1])

        if line.find(line5) != -1:
            arry5.append(line[-6:-1])

        if line.find(line6) != -1:
            arry6.append(line[-6:-1])

        if line.find(line7) != -1:
            arry7.append(line[-6:-1])

        if line.find(line8) != -1:
            arry8.append(line[-6:-1])

        if line.find(line9) != -1:
            arry9.append(line[-6:-1])

        if line.find(line10) != -1:
            arry10.append(line[-6:-1])

        if line.find(line11) != -1:
            arry11.append(line[-6:-1])

        if line.find(line12) != -1:
            arry12.append(line[-6:-1])

        if line.find("Epoch: [") != -1 and line.find("Total time") == -1: # line.find("Epoch: [0]") != -1
            lr_index = line.find("lr: ")
            lr_value = line[lr_index+4:lr_index+12]
            lr.append(lr_value)
            print(lr_value)

            loss_index = line.find("loss: ")
            loss_value = line[loss_index+6:loss_index+12]
            loss.append(loss_value)
            print(loss_value)


    fp.close()

    path, name = os.path.split(file_path)

    with open(path+"/out/"+name.split(".")[0]+"_out.txt", "w+") as fp2:
        #str_lr = convert_list2str(lr, "lr")
        #fp2.write(str_lr+"\n")

        # str_loss = convert_list2str(loss, "loss")
        # fp2.write(str_loss+"\n")

        str_line1 = convert_list2str(arry1, "AP_ALL")
        fp2.write(str_line1+"\n")

        str_line2 = convert_list2str(arry2, "AP_ALL_50")
        fp2.write(str_line2+"\n")

        str_line3 = convert_list2str(arry3, "AP_ALL_75")
        fp2.write(str_line3+"\n")

        str_line4 = convert_list2str(arry4, "AP_small")
        fp2.write(str_line4+"\n")

        str_line5 = convert_list2str(arry5, "AP_medium")
        fp2.write(str_line5+"\n")

        str_line6 = convert_list2str(arry6, "AP_large")
        fp2.write(str_line6+"\n")

        str_line7 = convert_list2str(arry7, "AR_All_1")
        fp2.write(str_line7+"\n")

        str_line8 = convert_list2str(arry8, "AR_All_10")
        fp2.write(str_line8+"\n")

        str_line9 = convert_list2str(arry9, "AR_All_100")
        fp2.write(str_line9+"\n")

        str_line10 = convert_list2str(arry10, "AR_small")
        fp2.write(str_line10+"\n")

        str_line11 = convert_list2str(arry11, "AR_medium")
        fp2.write(str_line11+"\n")

        str_line12 = convert_list2str(arry12, "AR_large")
        fp2.write(str_line12+"\n")

# 观察损失和准确率的变化 plot_metric(dfhistory,"loss") plot_metric(dfhistory,"auc")
def plot_metric(dfhistory, metric): 
    train_metrics = dfhistory[metric] 
    val_metrics = dfhistory['val_'+metric] 
    epochs = range(1, len(train_metrics) + 1) 
    plt.plot(epochs, train_metrics, 'bo--') 
    plt.plot(epochs, val_metrics, 'ro-') 
    plt.title('Training and validation '+ metric) 
    plt.xlabel("Epochs") 
    plt.ylabel(metric) 
    plt.legend(["train_"+metric, 'val_'+metric]) 
    plt.show() 

if __name__ == '__main__':

    file_root = "/Users/hubin/experimentsLog/bccd_experiments/"
    # file_path = file_root + "originmodel_voc.log"
    # read_log_file(file_path)

    file_path = file_root + "E2_just_objectnessLoss_new_bccd_valid.log"
    read_log_file(file_path)


