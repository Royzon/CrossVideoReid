import tensorflow as tf
import numpy as np
def IoU_calculator(box1, box2, wh=True,myself=False):
    """
    compute the iou of two boxes.
    Args:
        box1, box2: [xmin, ymin, xmax, ymax] (wh=False) or [xcenter, ycenter, w, h] (wh=True)
        wh: the format of coordinate.
    Return:
        iou: iou of box1 and box2.
    """
    if wh == False:
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2
    else:
        xmin1, ymin1 = int(box1[0]-box1[2]/2.0), int(box1[1]-box1[3]/2.0)
        xmax1, ymax1 = int(box1[0]+box1[2]/2.0), int(box1[1]+box1[3]/2.0)
        xmin2, ymin2 = int(box2[0]-box2[2]/2.0), int(box2[1]-box2[3]/2.0)
        xmax2, ymax2 = int(box2[0]+box2[2]/2.0), int(box2[1]+box2[3]/2.0)

    ## 获取矩形框交集对应的左上角和右下角的坐标（intersection）
    xx1 = np.max([xmin1, xmin2])
    yy1 = np.max([ymin1, ymin2])
    xx2 = np.min([xmax1, xmax2])
    yy2 = np.min([ymax1, ymax2])
    ## 计算两个矩形框面积
    area1 = (xmax1-xmin1) * (ymax1-ymin1) 
    area2 = (xmax2-xmin2) * (ymax2-ymin2)
    inter_area = (np.max([0, xx2-xx1])) * (np.max([0, yy2-yy1]))#计算交集面积
    iou = inter_area / (area1+area2-inter_area+1e-6)#计算交并比
    if myself==True:
        iou1 = inter_area/area1+1e-6
        iou2 = inter_area/area2+1e-6
        return iou1,iou2
    return iou

# tensorflow过慢暂时不使用

def IoU_calculator_tensorflow(x, y, w, h, l_x, l_y, l_w, l_h):
    """calaulate IoU
    Args:
      x: net predicted x
      y: net predicted y
      w: net predicted width
      h: net predicted height
      l_x: label x
      l_y: label y
      l_w: label width
      l_h: label height
    Returns:
      IoU
    """
    # convert to coner
    x_max = x + w/2
    y_max = y + h/2
    x_min = x - w/2
    y_min = y - h/2
 
    l_x_max = l_x + l_w/2
    l_y_max = l_y + l_h/2
    l_x_min = l_x - l_w/2
    l_y_min = l_y - l_h/2
    # calculate the inter
    inter_x_max = tf.minimum(x_max, l_x_max)
    inter_x_min = tf.maximum(x_min, l_x_min)
 
    inter_y_max = tf.minimum(y_max, l_y_max)
    inter_y_min = tf.maximum(y_min, l_y_min)
 
    inter_w = inter_x_max - inter_x_min
    inter_h = inter_y_max - inter_y_min
    
    inter = tf.cond(tf.logical_or(tf.less_equal(inter_w,0), tf.less_equal(inter_h,0)), 
                    lambda:tf.cast(0,tf.float32), 
                    lambda:tf.multiply(inter_w,inter_h))
    # calculate the union
    union = w*h + l_w*l_h - inter
    
    IoU = inter / union
    return IoU
#交集占第一个bbox的面积比例
def myself_iou_tensorflow(x, y, w, h, l_x, l_y, l_w, l_h):
    # convert to coner
    x_max = x + w/2
    y_max = y + h/2
    x_min = x - w/2
    y_min = y - h/2
    l_x_max = l_x + l_w/2
    l_y_max = l_y + l_h/2
    l_x_min = l_x - l_w/2
    l_y_min = l_y - l_h/2
    # calculate the inter
    inter_x_max = tf.minimum(x_max, l_x_max)
    inter_x_min = tf.maximum(x_min, l_x_min)
    inter_y_max = tf.minimum(y_max, l_y_max)
    inter_y_min = tf.maximum(y_min, l_y_min)
    inter_w = inter_x_max - inter_x_min
    inter_h = inter_y_max - inter_y_min
    inter = tf.cond(tf.logical_or(tf.less_equal(inter_w,0), tf.less_equal(inter_h,0)), 
                    lambda:tf.cast(0,tf.float32), 
                    lambda:tf.multiply(inter_w,inter_h))
    # calculate the union
    union1 = w*h
    union2 = l_w*l_h
    IoU1 = inter / union1
    IoU2 = inter / union2
    return IoU1,IoU2