#TODO rpn部分损失函数计算器改造  my_loss_computor ---> my_rpn--->MyFasterRCNN--->faster_rcnn_small_target
import torch
from torch.nn import functional as F
from torch import nn, Tensor
from torchvision.models.detection import _utils as det_utils
from torchvision.transforms import transforms as T
import torchvision.models.detection.roi_heads


batch_size_per_image=256
positive_fraction=0.5

fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image, positive_fraction
        )


#此处为rpn计算损失函数环节
def my_compute_loss( objectness, pred_bbox_deltas, labels, regression_targets, pred_bbox_fix_pix):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    Args:
        objectness (Tensor)
        pred_bbox_deltas (Tensor)
        labels (List[Tensor])
        regression_targets (List[Tensor])

    Returns:
        objectness_loss (Tensor)
        box_loss (Tensor)
    """
    #设置是否采用新的损失函数
    use_objectness_new_loss = True
    use_rpn_box_new_loss = False

    sampled_pos_inds, sampled_neg_inds = fg_bg_sampler(labels)
    sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
    sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]

    sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

    objectness = objectness.flatten()

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)
    if use_rpn_box_new_loss == True :
        #对rpn位置回归损失计算加入面积系数
        box_loss = smooth_l1_loss_with_pix_balance(
            pred_bbox_deltas[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            pred_bbox_fix_pix[sampled_pos_inds],
            beta=1 / 9,
            ) / (sampled_inds.numel())
    else:
        box_loss = F.smooth_l1_loss(
            pred_bbox_deltas[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1 / 9,
            reduction='sum',
        ) / (sampled_inds.numel())

    if use_objectness_new_loss==True:
        #对类别损失计算加入面积系数
        objectness_loss = my_binary_cross_entropy_with_logits(
            objectness[sampled_inds], labels[sampled_inds], pred_bbox_fix_pix[sampled_inds]
        )
    else:
        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness[sampled_inds], labels[sampled_inds]
        )

    return objectness_loss, box_loss

#改造head部分的损失计算
def compute_fastrcnn_loss(class_logits, box_regression, labels, regression_targets,box_regression_pix):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    Computes the loss for Faster R-CNN.

    Args:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)
        box_regression_pix (Tensor) 与box_regression对应的预测框真实坐标xyxy值

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """
    # 设置是否采用新的损失函数
    use_classify_new_loss = False
    use_box_new_loss = False

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)
    if use_classify_new_loss == True:
        classification_loss = my_cross_entropy(class_logits, labels, box_regression_pix)
    else:
        classification_loss = F.cross_entropy(class_logits, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

    if use_box_new_loss == True:
        box_loss = smooth_l1_loss_with_pix_balance(
            box_regression[sampled_pos_inds_subset, labels_pos],
            regression_targets[sampled_pos_inds_subset],
            box_regression_pix[sampled_pos_inds_subset],
            beta=1 / 9,
        )
        box_loss = box_loss / labels.numel()
    else:
        box_loss = F.smooth_l1_loss(
            box_regression[sampled_pos_inds_subset, labels_pos],
            regression_targets[sampled_pos_inds_subset],
            beta=1 / 9,
            reduction='sum',
        )
        box_loss = box_loss / labels.numel()

    return classification_loss, box_loss

#加入权重系数累计分类损失，因为是背景前景的二分类问题，因此用**_logits，内涵sigmoid转换
def my_binary_cross_entropy_with_logits(objectness, labels,pred_bbox_fix_pix) -> object:
    aeras = compute_aera(pred_bbox_fix_pix)
    penalty_fenzi = torch.mean(aeras)
    aeras_param = 2*penalty_fenzi/(aeras+penalty_fenzi)
    weight = aeras_param

    new_objectness_loss = F.binary_cross_entropy_with_logits(objectness, labels,weight)
    return new_objectness_loss

#加入pix_balance_weight调和的分类识别交叉熵损失函数
def my_cross_entropy(class_logits, labels, box_regression_pix):
    #计入面积系数，且整合到weight权重中，符合输入项是捆绑检测框大小的，设计成和面积+类别协同平衡
    aeras = compute_aera(box_regression_pix)
    penalty_fenzi = torch.mean(aeras)
    aeras_param = 2*penalty_fenzi/(aeras+penalty_fenzi)
    weight = aeras_param
    new_classification_loss = cross_entropy_with_pix_balance(class_logits, labels,weight)
    return new_classification_loss


def smooth_l1_loss_with_pix_balance(input, target,pred_bbox_fix_pix, beta=1./(3**2), reduce=True, normalizer=1.):
    diff = torch.abs(input - target)
    cond = diff < beta
    loss = torch.where(cond, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    aeras = compute_aera(pred_bbox_fix_pix)
    # compute the mean_aera value
    area_mean = torch.mean(aeras)
    pix_balance_weight = 2*area_mean/(aeras+area_mean)
    pix_balance_weight = pix_balance_weight.reshape(-1,1)

    #loss multiple  pix_balance_weight
    new_loss = loss * pix_balance_weight
    loss = new_loss
    if reduce:
        return torch.sum(loss) / normalizer
    return torch.sum(loss, dim=1) / normalizer

#input的值来计算面积，考虑输入来源的像素数目大小问题
def compute_aera(input):
    # 需要boxencoder后的原始坐标来计算像素面积
    input_width = input[:,2]-input[:,0]
    input_height = input[:,3]-input[:,1]
    aeras = input_width * input_height
    return aeras

def cross_entropy_with_pix_balance (object_score,object_label,pix_balance_weight):
    loss = 0
    input = F.softmax(object_score)
    input = torch.log(input)
    loss_fn = nn.NLLLoss()
    #get input_shape
    batch_size, class_num = input.size()
    #accumulate every box's cross_entropy loss
    weight = pix_balance_weight / batch_size
    for i in range(batch_size):
        batchloss = loss_fn(input[i], object_label[i])
        # each loss multiple box's pix_balance_weight
        batchloss = batchloss * weight[i]
        loss = loss + batchloss
    return loss

import numpy as np
def main():
    x = np.array([[1, 2, 3, 4, 5],  # 共三3样本，有5个类别
                  [1, 2, 3, 4, 5],
                  [1, 2, 3, 4, 5]]).astype(np.float32)
    y = np.array(
        [3, 2, 1])  # 这3个样本的标签分别是1,1,0即两个是第2类，一个是第1类。多类别这种，并不是one-hot编码，因此后面的损失计算是直接用对应标签索引到样本对应的正确置信度，直接求和得到batch的loss
    x = torch.from_numpy(x)
    y = torch.from_numpy(y).long()

    weight = np.array([1,1,1]).astype(np.float32)
    weight = torch.from_numpy(weight)
    loss = F.cross_entropy(x, y)
    print(loss)
    batchloss = cross_entropy_with_pix_balance(x,y,weight)
    print(batchloss)


if __name__ == "__main__":
    main()

