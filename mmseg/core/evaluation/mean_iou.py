import numpy as np
import pdb


def intersect_and_union(pred_label, label, num_classes, ignore_index):
    """Calculate intersection and Union.

    Args:
        pred_label (ndarray): Prediction segmentation map
        label (ndarray): Ground truth segmentation map
        num_classes (int): Number of categories
        ignore_index (int): Index that will be ignored in evaluation.

     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes
         ndarray: The union of prediction and ground truth histogram on all
             classes
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """

    if len(label.shape)!=2:
        print('[Warning] cal miou: label!=2', end=' ')
        print(label.shape)
        #return [0, 0], [1e-9, 1e-9], 0, 0
        return 0, 0, 0, 0
    assert pred_label.shape == label.shape
    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect, _ = np.histogram(
        intersect, bins=np.arange(num_classes + 1))
    area_pred_label, _ = np.histogram(
        pred_label, bins=np.arange(num_classes + 1))
    area_label, _ = np.histogram(label, bins=np.arange(num_classes + 1))
    area_union = area_pred_label + area_label - area_intersect

    return area_intersect, area_union, area_pred_label, area_label


def mean_iou(results, gt_seg_maps, num_classes, ignore_index):
    """Calculate Intersection and Union (IoU)

    Args:
        results (list[ndarray]): List of prediction segmentation maps
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps
        num_classes (int): Number of categories
        ignore_index (int): Index that will be ignored in evaluation.

     Returns:
         float: Overall accuracy on all images.
         ndarray: Per category accuracy, shape (num_classes, )
         ndarray: Per category IoU, shape (num_classes, )
    """

    num_imgs = len(results)
    num_mask_layer = results[0].shape[0]
    #print('num_mask_layer', num_mask_layer)
    assert len(gt_seg_maps) == num_imgs
    total_area_intersect = np.zeros((num_classes, ), dtype=np.float)
    total_area_union = np.zeros((num_classes, ), dtype=np.float)
    total_area_pred_label = np.zeros((num_classes, ), dtype=np.float)
    total_area_label = np.zeros((num_classes, ), dtype=np.float)
    for i in range(num_imgs):
        #with open('tmp.txt', 'a') as fout:
        #    print(i, end='', file=fout)
        
        max_iou = -1.0
        res = []
        cur_iou_list = []
        for j in range(num_mask_layer):
            #print('pred.shape and gt.shape', results[i][j, :, :].shape, gt_seg_maps[i].shape)
            area_intersect, area_union, area_pred_label, area_label = \
                intersect_and_union(results[i][j, :, :], gt_seg_maps[i], num_classes,
                                    ignore_index=ignore_index)
            if type(area_union) is int:
                max_iou = 0.0
                res = [area_intersect, area_union, area_pred_label, area_label]
                break
            else:
                cur_iou = area_intersect * 1.0 / area_union
                #print('iou:', cur_iou)
                cur_iou_list.append((cur_iou, [area_intersect, area_union, area_pred_label, area_label]))
                if cur_iou[1] > max_iou:
                    max_iou = cur_iou[1]
                    res = [area_intersect, area_union, area_pred_label, area_label]
                
        area_intersect, area_union, area_pred_label, area_label = res
        
        #print('max_iou:', max_iou)
        #print('res:', res)
        #for it in cur_iou_list:
        #    print(it)
        #print('end')
                
        #with open('tmp.txt', 'a') as fout:
        #    print('', area_intersect / (area_union+1e-9), file=fout)
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    all_acc = total_area_intersect.sum() / total_area_label.sum()
    acc = total_area_intersect / total_area_label
    iou = total_area_intersect / total_area_union

    return all_acc, acc, iou, total_area_intersect, total_area_union
