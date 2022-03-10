import xml.etree.ElementTree as ET
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

import torch
from torchvision import ops
import torch.nn.functional as F
import torch.optim as optim

def get_req_output(model_output, n_anchors, indices_all):
    batch_size = model_output.size(dim=0)
    model_output_flat = flatten_model_output(model_output, n_anchors)
    
    req_output = []
    for i in range(batch_size):
        indices = indices_all[i]
        model_output_i = model_output_flat[i]
        
        req_output.append(model_output_i[indices])
        
    return req_output
        
def flatten_model_output(model_output, n_anchors):
    B, _, Hmap, Wmap = model_output.size()
    # reshape as [batch_size, n_anchors, (score or offsets), height, width]
    flat_model_output = model_output.reshape(B, n_anchors, -1, Hmap, Wmap) 
    
    B, A, D, H, W = flat_model_output.shape
    flat_model_output = flat_model_output.permute(0, 1, 3, 4, 2).contiguous().view(B, -1, D)
    return flat_model_output

def parse_annotation(annotation_path, img_size):
    '''
    Traverse the xml tree, get the annotations, and resize them to the scaled image size
    '''
    img_h, img_w = img_size

    with open(annotation_path, "r") as f:
        tree = ET.parse(f)

    root = tree.getroot()  
    
    img_path = None
    # get image path
    for object_ in root.findall('path'):
        img_path = object_.text
      
    # get raw image size    
    for object_ in root.findall('size'):
        orig_w = int(object_.find("width").text)
        orig_h = int(object_.find("height").text)
            
    # get bboxes        
    groundtruth_boxes = []
    for object_ in root.findall('object/bndbox'):
        xmin = int(object_.find("xmin").text)
        ymin = int(object_.find("ymin").text)
        xmax = int(object_.find("xmax").text)
        ymax = int(object_.find("ymax").text)
        
        # rescale bboxes
        bbox = torch.Tensor([xmin, ymin, xmax, ymax])
        bbox[[0, 2]] = bbox[[0, 2]] * img_w/orig_w
        bbox[[1, 3]] = bbox[[1, 3]] * img_h/orig_h
        
        groundtruth_boxes.append(bbox.tolist())
        
    # get classes        
    groundtruth_classes = []
    for object_ in root.findall('object/name'):
        groundtruth_classes.append(object_.text)
                
    return torch.Tensor(groundtruth_boxes), img_path, groundtruth_classes

def calc_gt_offsets_all(batch_size, pos_anc_ind_all, gt_bbox_mapping_all, anc_boxes_all):
    gt_offsets_pos = []
    anc_boxes_flat = anc_boxes_all.flatten(start_dim=1, end_dim=-2)
    
    for i in range(batch_size):
        pos_anc_ind = pos_anc_ind_all[i]
        pos_anc_coords = anc_boxes_flat[i][pos_anc_ind]
        gt_bbox_mapping = gt_bbox_mapping_all[i]
        
        gt_offsets = calc_gt_offsets(pos_anc_coords, gt_bbox_mapping)
        gt_offsets_pos.append(gt_offsets)
        
    return torch.cat(gt_offsets_pos)

def calc_gt_offsets(pos_anc_coords, gt_bbox_mapping):
    pos_anc_coords = ops.box_convert(pos_anc_coords, in_fmt='xyxy', out_fmt='cxcywh')
    gt_bbox_mapping = ops.box_convert(gt_bbox_mapping, in_fmt='xyxy', out_fmt='cxcywh')

    gt_cx, gt_cy, gt_w, gt_h = gt_bbox_mapping[:, 0], gt_bbox_mapping[:, 1], gt_bbox_mapping[:, 2], gt_bbox_mapping[:, 3]
    anc_cx, anc_cy, anc_w, anc_h = pos_anc_coords[:, 0], pos_anc_coords[:, 1], pos_anc_coords[:, 2], pos_anc_coords[:, 3]

    tx_ = (gt_cx - anc_cx)/anc_w
    ty_ = (gt_cy - anc_cy)/anc_h
    tw_ = torch.log(gt_w / anc_w)
    th_ = torch.log(gt_h / anc_h)

    return torch.stack([tx_, ty_, tw_, th_], dim=-1)

# loss helper functions
def calc_conf_loss(conf_scores_pos, conf_scores_neg):
    target_pos = torch.ones_like(conf_scores_pos)
    target_neg = torch.zeros_like(conf_scores_neg)
    
    target = torch.cat((target_pos, target_neg))
    inputs = torch.cat((conf_scores_pos, conf_scores_neg))
    
    loss = F.binary_cross_entropy_with_logits(inputs, target)
    
    return loss

def calc_bbox_reg_loss(gt_offsets, reg_offsets_pos):
    assert gt_offsets.size() == reg_offsets_pos.size()
    loss = F.smooth_l1_loss(reg_offsets_pos, gt_offsets) 
    return loss

# anchors helper functions
def gen_anc_centers(out_size):
    out_h, out_w = out_size
    
    # define mappings from feature space to image
    anc_pts_x = torch.arange(0, out_w) + 0.5
    anc_pts_y = torch.arange(0, out_h) + 0.5
    
    return anc_pts_x, anc_pts_y

def project_bboxes(bboxes, width_scale_factor, height_scale_factor, mode='a2p'):
    assert mode in ['a2p', 'p2a']
    
    batch_size = bboxes.size(dim=0)
    proj_bboxes = bboxes.clone().reshape(batch_size, -1, 4)
    invalid_bbox_mask = (proj_bboxes == -1) # indicating padded bboxes
    
    if mode == 'a2p':
        # activation map to pixel image
        proj_bboxes[:, :, [0, 2]] *= width_scale_factor
        proj_bboxes[:, :, [1, 3]] *= height_scale_factor
    else:
        # pixel image to activation map
        proj_bboxes[:, :, [0, 2]] /= width_scale_factor
        proj_bboxes[:, :, [1, 3]] /= height_scale_factor
        
    proj_bboxes.masked_fill_(invalid_bbox_mask, -1) # fill padded bboxes back with -1
    proj_bboxes.resize_as_(bboxes)
    
    return proj_bboxes

def generate_proposals(anchors, offsets):
    
    # flatten anchors and offsets
    anchors_flat = anchors.flatten(start_dim=1, end_dim=-2)
    offsets_flat = offsets.flatten(start_dim=1, end_dim=-2)
   
    # change format of the anchor boxes from 'xyxy' to 'cxcywh'
    anchors_flat = ops.box_convert(anchors_flat, in_fmt='xyxy', out_fmt='cxcywh')

    # apply offsets to anchors to create proposals
    proposals_ = torch.zeros_like(anchors_flat)
    proposals_[:,:,0] = anchors_flat[:,:,0] + offsets_flat[:,:,0]*anchors_flat[:,:,2]
    proposals_[:,:,1] = anchors_flat[:,:,1] + offsets_flat[:,:,1]*anchors_flat[:,:,3]
    proposals_[:,:,2] = anchors_flat[:,:,2] * torch.exp(offsets_flat[:,:,2])
    proposals_[:,:,3] = anchors_flat[:,:,3] * torch.exp(offsets_flat[:,:,3])

    # change format of proposals back from 'cxcywh' to 'xyxy'
    proposals = ops.box_convert(proposals_, in_fmt='cxcywh', out_fmt='xyxy')

    return proposals

def gen_anc_base(anc_pts_x, anc_pts_y, anc_scales, anc_ratios, out_size):
    n_anc_boxes = len(anc_scales) * len(anc_ratios)
    anc_base = torch.zeros(1, anc_pts_x.size(dim=0) \
                              , anc_pts_y.size(dim=0), n_anc_boxes, 4) # shape - [1, Hmap, Wmap, n_anchor_boxes, 4]
    
    for ix, xc in enumerate(anc_pts_x):
        for jx, yc in enumerate(anc_pts_y):
            anc_boxes = torch.zeros((n_anc_boxes, 4))
            c = 0
            for i, scale in enumerate(anc_scales):
                for j, ratio in enumerate(anc_ratios):
                    w = scale * ratio
                    h = scale
                    
                    xmin = xc - w / 2
                    ymin = yc - h / 2
                    xmax = xc + w / 2
                    ymax = yc + h / 2

                    anc_boxes[c, :] = torch.Tensor([xmin, ymin, xmax, ymax])
                    c += 1

            anc_base[:, ix, jx, :] = ops.clip_boxes_to_image(anc_boxes, size=out_size)
            
    return anc_base

def get_req_anchors(batch_size, anc_boxes_all, gt_bboxes_all, pos_thresh, neg_thresh):
    
    # flatten anchor boxes
    anc_boxes_flat = anc_boxes_all.reshape(batch_size, -1, 4)
    # get total anchor boxes for a single image
    tot_anc_boxes = anc_boxes_flat.size(dim=1)
    
    # create a placeholder to compute IoUs amongst the boxes
    ious_all = torch.zeros((batch_size, tot_anc_boxes, gt_bboxes_all.size(dim=1)))

    pos_anc_ind_all = [] # positive anchor indices for all the images
    neg_anc_ind_all = [] # negative anchor indices for all the images

    # compute IoU of the anc boxes with the gt boxes for all the images
    for i in range(batch_size):
        gt_bboxes = gt_bboxes_all[i]
        anc_boxes = anc_boxes_flat[i]
        ious_all[i, :] = ops.box_iou(anc_boxes, gt_bboxes)

        # get positive anchor indices
        pos_anc_ind_a = ious_all[i, :].argmax(dim=0) # condition 1
        pos_anc_ind_a = pos_anc_ind_a[pos_anc_ind_a != 0]
        pos_anc_ind_b = torch.where(ious_all[i, :] > pos_thresh)[0] # condition 2
        # combine condition 1 & 2
        pos_anc_ind = torch.unique(torch.cat((pos_anc_ind_a, pos_anc_ind_b)))

        # get negative anchor indices
        neg_anc_ind = torch.all((ious_all[i, :] < neg_thresh) & (ious_all[i, :] >= 0), dim=1)
        neg_anc_ind = torch.where(neg_anc_ind)[0]

        # sample -ve anchors to avoid data imbalance
        neg_anc_ind_sample = neg_anc_ind[torch.randint(0, neg_anc_ind.shape[0], \
                                                          (pos_anc_ind.shape[0],))]

        pos_anc_ind_all.append(pos_anc_ind)
        neg_anc_ind_all.append(neg_anc_ind_sample)
        
    return pos_anc_ind_all, neg_anc_ind_all, ious_all

def map_gt(batch_size, pos_anc_ind_all, gt_bboxes_all, gt_classes_all, ious_all):
    # map each positive anchor box to its corresponding ground truth box
    gt_bbox_mapping_all = []
    gt_classes_mapping_all = []
    
    for i in range(batch_size):
        gt_bboxes = gt_bboxes_all[i]
        gt_classes = gt_classes_all[i]
        
        pos_anc_ind = pos_anc_ind_all[i]
        gt_bbox_mapping_idx = ious_all[i, :][pos_anc_ind].argmax(dim=1)
        
        gt_bbox_mapping = torch.Tensor([gt_bboxes[k].tolist() for k in gt_bbox_mapping_idx])
        gt_classes_mapping = torch.Tensor([gt_classes[k] for k in gt_bbox_mapping_idx])

        gt_bbox_mapping_all.append(gt_bbox_mapping)
        gt_classes_mapping_all.append(gt_classes_mapping)
        
    return gt_bbox_mapping_all, gt_classes_mapping_all

# training functions
def training_loop(model, learning_rate, train_dataloader, n_epochs):
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    loss_list = []
    
    for i in tqdm(range(n_epochs)):
        total_loss = 0
        n_batches = 0
        for img_batch, gt_bboxes_batch, gt_classes_batch in train_dataloader:
            
            # forward pass
            loss = model(img_batch, gt_bboxes_batch, gt_classes_batch)
            
            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        loss_per_batch = total_loss / n_batches
        loss_list.append(loss_per_batch)
        
    return loss_list

# plotting helper functions
def display_img(img_data, fig, axes):
    for i, img in enumerate(img_data):
        if type(img) == torch.Tensor:
            img = img.permute(1, 2, 0).numpy()
        axes[i].imshow(img)
    
    return fig, axes

def display_bbox(bboxes, fig, ax, in_format='xyxy', color='w', line_width=3):
    if type(bboxes) == np.ndarray:
        bboxes = torch.from_numpy(bboxes)
    # convert boxes to xywh format
    bboxes = ops.box_convert(bboxes, in_fmt=in_format, out_fmt='xywh')
    for box in bboxes:
        x, y, w, h = box.numpy()
        rect = patches.Rectangle((x, y), w, h, linewidth=line_width, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
    return fig, ax

def display_grid(x_points, y_points, fig, ax, special_point=None):
    # plot grid
    for x in x_points:
        for y in y_points:
            ax.scatter(x, y, color="yellow", marker='+')
            
    # plot a special point we want to emphasize on the grid
    if special_point:
        x, y = special_point
        ax.scatter(x, y, color="red", marker='+')
        
    return fig, ax