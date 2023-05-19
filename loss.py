import torch
import torchvision.ops as ops

from torchvision.ops.boxes import _box_inter_union

def GIoU_loss(input_boxes, target_boxes, eps=1e-7):
   
    inter, union = _box_inter_union(input_boxes, target_boxes)
    iou = inter / union

   

    # area of the smallest enclosing box
    min_box = torch.min(input_boxes, target_boxes)
    max_box = torch.max(input_boxes, target_boxes)
    area_c = (max_box[:, 2] - min_box[:, 0]) * (max_box[:, 3] - min_box[:, 1])

    giou = iou - ((area_c - union) / (area_c + eps))

    loss = 1 - giou

    #return loss.sum()
    return loss




def loss_function(predicted_box, target_box,  lambda_l1=1.0, lambda_giou=1.0):
    # Calculate the L1 loss
    l1_loss = torch.abs(predicted_box - target_box).sum(dim=1)

    giou_loss= GIoU_loss(predicted_box,target_box )
    
    # Calculate the total loss
    total_loss = lambda_l1 * l1_loss + lambda_giou * giou_loss
    
    return total_loss.mean()
  

    

