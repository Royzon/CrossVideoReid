import torch
torch.nn.Module.dump_patches = True
import os
import shutil
import random
import time
import cv2
import numpy as np
from yacs.config import CfgNode as CN

import sys
import torch
import copy
import queue
import glob
import math
import random
import itertools
import threading
from scipy.spatial.distance import pdist
from scipy.optimize import linear_sum_assignment
from utils.datasets import LoadStreams,LoadImages
from utils.utils import torch_utils,google_utils,non_max_suppression,Path,scale_coords,plot_one_box,platform,xyxy2xywh
from CrossVideo.reid import CNN_REID
def display(bbox_list,im,identities,dths,color,t):
        #print(len(out_list))
        for bbox,id_,dth,color_ in zip(bbox_list,identities,dths,color):
            #(x1, y1, x2, y2) = bbox
            (x, y, w, h) = bbox
            (x1, y1, x2, y2)=int(x-w/2), int(y-h/2), int(x+w/2), int(y+h/2)
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, color_, 255-color_), 2)
            #cv2.putText(im, ('%.2f'%c), (x2+10, y1+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 250,0), 1)
            cv2.putText(im, ('%s'%id_), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, color_, 255-color_), 1)
            cv2.putText(im, ('%.2f'%dth), (x2, y1+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, color_, 255-color_), 1)
            cv2.circle(im, (int(x),int(y)), 1, (0, 255, 0), 2)
        fps = 1/(time.time()-t)
        cv2.putText(im, ('%.2f'%fps), (1800, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow('efficientdet', im)
        return im
def yolov5_detect(YOLOV5_CFG,reid):
    device = torch_utils.select_device(YOLOV5_CFG.device)
    model = torch.load(YOLOV5_CFG.weights, map_location=device)['model']
    #model = torch.load(YOLOV5_CFG.weights, map_location=device)
    model.to(device).eval()
    names = model.names if hasattr(model, 'names') else model.modules.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    
    if os.path.isfile(YOLOV5_CFG.source) is True:
        cap = cv2.VideoCapture(YOLOV5_CFG.source)
        w0,h0,fps,max_index = int(cap.get(3)),int(cap.get(4)),int(cap.get(5)),int(cap.get(7))
        [w1,h1] = YOLOV5_CFG.im_size
        if YOLOV5_CFG.output is not False:
            out = cv2.VideoWriter(YOLOV5_CFG.output, cv2.VideoWriter_fourcc(*'XVID'), fps, (w0,h0))
        for i in range(max_index):
            re,im = cap.read()
            net_input = np.transpose(cv2.resize(im, (w1,h1))/255,(2,0,1)).reshape((-1,3,h1,w1))
            net_input = torch.from_numpy(net_input).to(device).type(torch.float32)
            t = time.time()
            pred = model(net_input, augment=YOLOV5_CFG.augment)[0]
            pred = non_max_suppression(pred, YOLOV5_CFG.conf_thres, YOLOV5_CFG.iou_thres, fast=True, classes=YOLOV5_CFG.classes, agnostic=YOLOV5_CFG.agnostic_nms)
            if pred is None or pred[0] is None:
                cv2.imshow('A',im)
                if YOLOV5_CFG.output is not False:
                    out.write(im)
                continue
            bboxes,confs,cats = pred[0][:,:4].cpu().detach().numpy(),pred[0][:,4].cpu().detach().numpy(),pred[0][:,5].cpu().detach().numpy()
            bboxes[:,[0,2]],bboxes[:,[1,3]]=bboxes[:,[0,2]]*(w0/w1),bboxes[:,[1,3]]*(h0/h1)
            list_bbox,list_conf=[],[]
            for bbox,conf,cat in zip(bboxes.astype(np.int),confs,cats.astype(np.int)):
                if (names[cat]=='person'):
                    #p_min,p_max = (bbox[0],bbox[1]),(bbox[2],bbox[3])
                    #im = cv2.rectangle(im, p_min, p_max, (255,0,123), 1, cv2.LINE_AA)
                    #im = cv2.putText(im, '%s %.2f'%(names[cat],conf), p_min, cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
                    (x1, y1, x2, y2)=bbox
                    bbox_=(int(abs((x1+x2)*0.5)),int(abs((y1+y2)*0.5)),int(abs(x1-x2)),int(abs(y1-y2)))
                    list_bbox.append(bbox_)
                    list_conf.append(conf)
            #CrossVideo Reid
            out_bboxs,identities,confidence ,color,frame = reid.cross_video(list_bbox,list_conf,im)
            display(out_bboxs,im,identities,confidence,color,t)
            t=time.time()
            if YOLOV5_CFG.output is not False:
                out.write(im)
            if cv2.waitKey(1)&0xff==ord('q'):
                break
        cap.release()
        if YOLOV5_CFG.output is not False:
            out.release()
        cv2.destroyAllWindows()
if __name__ == '__main__':
    YOLOV5_CFG = CN()
    YOLOV5_CFG.agnostic_nms = False
    YOLOV5_CFG.augment      = False
    YOLOV5_CFG.classes      = False
    YOLOV5_CFG.device       = '0'

    YOLOV5_CFG.weights = 'weights/yolov5l.pt'
    YOLOV5_CFG.source = './testvideo/crossvideo_four.mp4'
    name = time.strftime('%Y.%m.%d',time.localtime(time.time()))
    YOLOV5_CFG.output = './savevideo/'+name+'.avi'


    YOLOV5_CFG.save_npz     = False
    YOLOV5_CFG.conf_thres   = 0.3
    YOLOV5_CFG.iou_thres    = 0.3
    YOLOV5_CFG.im_size      = [640,512]
    YOLOV5_CFG.freeze()

    Reid=CNN_REID(
                lib_dth=0.01,
                passage_dth=0.02,
                reid_dth=0.03,
                forget=10,
                kalman=False)
        #arg0:camera id
        #arg1:库添加的阈值 
        #arg2:切换状态的阈值 
        #arg3:分辨不同人的阈值 
        #arg4:状态变为隐藏的帧数 
        #arg5:kalman开启的阀门
    yolov5_detect(YOLOV5_CFG,Reid)