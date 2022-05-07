
import cv2
from bccd_cocostyle_dataloader import get_bccd_cocostyle_loader
font = cv2.FONT_HERSHEY_SIMPLEX

bccd_names = {'0': 'background', '1': 'Platelets', '2': 'RBC', '3': 'WBC'}

# 创建 dataloader
train_data_loader,test_data_loader = get_bccd_cocostyle_loader()



# 可视化
for imgs, target in test_data_loader:
    for i in range(len(imgs)):
        bboxes = []
        ids = []
        img = imgs[i]
        target_ = target[i] #原始值绘图
        boxes = target_['boxes'].numpy().tolist()
        labels = target_['labels'].numpy().tolist()
        for box in boxes:
            bboxes.append([box[0],
                           box[1],
                           box[0] + box[2],
                           box[1] + box[3]
                           ])
        for label in labels:
            ids.append(label)

        img = img.permute(1, 2, 0).numpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for box, id_ in zip(bboxes, ids):
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])

            class_name = bccd_names.get(str(id_))
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
            cv2.putText(img, text=class_name, org=(x1 + 5, y1 + 5), fontFace=font, fontScale=1,
                        thickness=2, lineType=cv2.LINE_AA, color=(0, 255, 0))
        cv2.imshow('test', img)
        cv2.waitKey()

