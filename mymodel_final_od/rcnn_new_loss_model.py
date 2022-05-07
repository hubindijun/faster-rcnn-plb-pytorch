import torchvision
import MyFasterRCNN as MyFasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
anchor_generator = AnchorGenerator(
                anchor_sizes, aspect_ratios
            )

roi_pooler = torchvision.ops.MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],
                output_size=7,
                sampling_ratio=2)


#自定义模型
def get_my_model_pretarined(num_classes = 91,pretrained=True):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = MyFasterRCNN.fasterrcnn_resnet50_fpn(pretrained=pretrained,
                                                 rpn_anchor_generator=anchor_generator,
                                                 box_roi_pool=roi_pooler)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = MyFasterRCNN.FastRCNNPredictor(in_features, num_classes)
    return model

def get_origin_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model


def get_model(num_classes=91, pretrained=True, is_origin=False):
    if is_origin:
        return get_origin_model(num_classes)
    else:
        return get_my_model_pretarined(num_classes, pretrained)
