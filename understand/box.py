import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torchvision.ops as ops
from torchvision.ops.boxes import _box_inter_union

def GIoU_loss(input_boxes, target_boxes, eps=1e-7):
   
    inter, union = _box_inter_union(input_boxes, target_boxes)
    iou = inter / union

    print("inter", inter)
    print("##############")
    print("union", union)


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
  






# Define the coordinates of the two boxes
box1 = [ 0.0000, 231.7800, 285.8300, 109.9400]
#box2 = [288.2601, 150.3590, 400.0191, 200.4614]
box2 = [247.6502, 132.4887, 275.4772, 183.2540]
 

# Create a figure and axes
fig, ax = plt.subplots()

# Create patches for the boxes
rect1 = patches.Rectangle((box1[0], box1[1]), box1[2]-box1[0], box1[3]-box1[1], linewidth=1, edgecolor='r', facecolor='none')
rect2 = patches.Rectangle((box2[0], box2[1]), box2[2]-box2[0], box2[3]-box2[1], linewidth=1, edgecolor='b', facecolor='none')
# Add the patches to the axes
ax.add_patch(rect1)
ax.add_patch(rect2)

# Set the x and y axis limits
ax.set_xlim(0, 400)
ax.set_ylim(0, 400)

# Set the aspect ratio to equal
ax.set_aspect('equal')

# Show the plot
plt.show()
