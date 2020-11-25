import os
import time
import cv2
import sys
import copy
import glob
import math
import random
import itertools
import threading
import numpy as np
from tqdm import tqdm
sys.path.append('..')
from CrossVideo.iou import *
from CrossVideo.Huangary import TaskAssignment
from CrossVideo.kalman import KalmanFilter
from CrossVideo.feature_extractor import Extractor
from collections import defaultdict

class CNN_deepsort:
    def __init__(self):
        self.extractor = Extractor("./CrossVideo/checkpoint/ckpt.t7", use_cuda=True)
    def compute_feature(self,im):
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        #print(im.shape)
        feature = self.extractor([im])
        #print(feature.shape)
        return feature

class CNN_REID:
    def __init__(self,cameras,lib_dth=0.01,passage_dth=0.02,reid_dth=0.03,forget=10,kalman=False):
        ''' arg0:camera ID arg1:库添加的阈值 arg2:切换状态的阈值 arg3: 分辨不同人的阈值 arg4:状态变为隐藏的帧数 arg5:kalman开启的阀门'''
        self.camera_ids=cameras
        self.frame=0
        self.color_old=0
        self.flag_frame=[]
        self.flag_line=[]
        self.add_names=[]
        self.lib_dth = 0.01 #库添加控制阈值
        self.passage_dth=0.02#通过的阈值
        self.reid_dth=0.03 #分辨不同人的阈值
        self.forget=10  #状态切换帧数
        #匈牙利算法
        self.task = TaskAssignment()
        #CNN的feature extractor 
        self.CNN = CNN_deepsort()
        #LIB保存 
        #公有参数
        self.save={'000':{'camera':{},'lib':[],'frame':0,'passage':False,'hold':False}}
        #每个摄像头拥有的私有参数
        for camera_id in self.camera_ids:
            self.save['000']['camera'][camera_id]={ 'bbox':[],
                                                    'center':[],
                                                    'kalman':[],
                                                    'cross':False,
                                                    'overlap':False,
                                                    'overlap_id':[]}
        #卡尔曼跟踪
        self.kalman=[]
        self.kalman_switch=kalman
    #*
    #*
    #*      计算集合
    #*
    #*
    
    #计算 余弦距离
    def compute_cos(self,x,y):
        x_,y_=x.flatten(),y.flatten()
        dist =1- abs(np.dot(x_,y_)/(np.linalg.norm(x_)*np.linalg.norm(y_)))      
        return abs(dist)
    #计算 欧氏距离
    def compute_dis(self,x,y):
         return np.linalg.norm( x - y )
    
     #计算两点近距离公式 xyxy
    def distEclud(self,veA,vecA,veB,vecB):
        lossA=veB-veA
        lossB=vecB-vecA
        return math.sqrt(pow(lossA,2)+pow(lossB,2))
    
     #输入图片计算，输出Feature
    def compute_feature_cnn(self,im):
        feature = self.CNN.compute_feature(im)
        return feature
    
    #找出list中same values
    def list_duplicates(self,seq):
        tally = defaultdict(list)
        for i,item in enumerate(seq):
            tally[item].append(i)
        return ((key,locs) for key,locs in tally.items() 
                                if len(locs)>1)
    #计算IOU
    def compute_IOU(self,bbox1,bbox2):
        iou = IoU_calculator(bbox1,bbox2,wh=True,myself=False)
        if iou ==0:
            (x1,y1,w1,h1)=bbox1
            (x2,y2,w2,h2)=bbox2
            dis = self.distEclud(x1,y1,x2,y2)
            if (dis/(w1+w2))<0.5:
                iou=1
        return iou
    #计算交集与本身面积的IOU
    def compute_IOU_myself(self,bbox1,bbox2):
        iou1,iou2 = IoU_calculator(bbox1,bbox2,wh=True,myself=True)
        if iou1==0 or iou2 ==0:
            (x1,y1,w1,h1)=bbox1
            (x2,y2,w2,h2)=bbox2
            dis = self.distEclud(x1,y1,x2,y2)
            if (dis/(w1+w2))<0.5:
                iou1,iou2=1,1
        return iou1,iou2
    #合并bbox
    def combine_bbox(self,bbox1,bbox2):
        (xi,yi,wi,hi)=bbox1
        (xj,yj,wj,hj)=bbox2
        xmin = min((xi-wi/2),(xj-wj/2))
        ymin = min((yi-hi/2),(yj-hj/2))
        xmax = max((xi+wi/2),(xj+wj/2))
        ymax = max((yi+hi/2),(yj+hj/2))
        x,y,w,h=(xmin+xmax)/2,(ymin+ymax)/2,abs(xmax-xmin),abs(ymax-ymin)
        bbox = (x,y,w,h)
        return bbox
    
    #*                  
    #  *  Kalman 函数 
    #*  

    def kalman_bbox_update(self,id_,bbox,camera_id):
        self.save[id_]['camera'][camera_id]['bbox']=bbox
        kalman_id=int(id_)-1
        self.kalman[kalman_id].update(bbox)
        x,y=self.kalman[kalman_id].get()
        self.save[id_]['camera'][camera_id]['kalman']=[x,y]
        return (int(x),int(y))
    #*                  
    #  *  ID管理函数集   
    #*                  
    
    def clear_id(self):
        sself.save={'000':{},'lib':[],'frame':0,'passage':False}

    def manage_lib(self,frame):
        for k,v in list(self.save.items()):
            if k=='000' :#or v[camera_id]['overlap']==True:#or v['passage']==False:
                continue
            lib = v['lib']
            frame_lib=v['frame']
            if frame-frame_lib>self.forget:
                self.save[k]['frame']=frame
                #self.save[k]['lib'].pop(0)
                #self.save[k]['lib'].pop(-1)
                if len(lib)<10:
                    self.dele_lib_id(k)
                if  k not in self.save:
                    continue
                if self.save[k]['passage']==True:
                    for camera_id in self.save[k]['camera'].keys():
                        center =self.save[k]['camera'][camera_id]['center']
                        for lib_ in self.save[k]['lib']:
                            center=center*0.9+lib_*0.1
                        #self.save[k]['lib'].pop(0)
                        self.passage_off(k,center,camera_id)
                        #self.flag_frame=[]
    def add_lib(self,feature,bbox,frame,camera_id,passage=False):
        #print('000',self.save['000'])
        save = copy.deepcopy(self.save['000'])
        for n in  self.save.keys():
            id_=n
        if self.kalman_switch:   
            self.kalman.append(KalmanFilter(bbox))
            self.kalman[(int(id_))].update(bbox)
            x,y=self.kalman[(int(id_))].get()
            save['camera'][camera_id]['kalman']=[x,y]
            
        self.id_new=('%03d'%(int(id_)+1))
        for name in self.camera_ids:
            save['camera'][name]['center']=feature
        save['camera'][camera_id]['bbox']=bbox
        #save['camera'][camera_id]['center']=feature
        save['lib']=[feature]
        save['frame']=frame
        save['passage']=passage
        self.save[self.id_new]=save
        #print('000',self.save['000'])
        print('find id %s'%self.id_new)
        return self.id_new
    
    def add_lib_overlab(self,feature,bbox,frame,list_ids,camera_id):
        save = copy.deepcopy(self.save['000'])
        for n in  self.save.keys():
            id_=n
        self.id_new=('%03d'%(int(id_)+1))

        if self.kalman_switch:
            self.kalman.append(KalmanFilter(bbox))
            self.kalman[(int(id_))].update(bbox)
            x,y=self.kalman[(int(id_))].get()
            save['camera'][camera_id]['kalman']=[x,y]
        
        save['camera'][camera_id]['overlap']=True
        save['camera'][camera_id]['overlap_id']=list_ids
        save['camera'][camera_id]['bbox']=bbox

        for name in self.camera_ids:
            save['camera'][name]['center']=feature
        #save['camera'][camera_id]['center']=feature
        save['lib']=[feature]
        save['hold']=True
        save['frame']=frame
        save['passage']=True
        self.save[self.id_new]=save
        print('find overlap id %s'%self.id_new)
        return self.id_new    
        
    def add_id(self,feature,bbox,camera_id,frame,state='None'):   
        #print(len(self.save))
        if frame!=0 and len(self.save)!=1:
            if state=='None':
                id_ = self.check_lib_passage(feature)
                if id_ == ' ':
                    self.flag_line=[]
                    self.add_names=[]
                    if self.id_new in self.save:
                        if frame-self.save[self.id_new]['frame']>1:
                            self.flag_frame=[]
                    else:
                        self.flag_frame=[]
                    if len(self.flag_frame)==0:
                        self.id_new = self.add_lib(feature,bbox,frame,camera_id)
                    self.flag_frame.append(frame)
                    self.save[self.id_new]['frame']=frame
                    #self.update_frame
                    if len(self.flag_frame)>=5:
                        if np.mean(self.flag_frame)==self.flag_frame[2]:
                            self.paassag_on(feature,bbox,frame,self.id_new,camera_id)
                            self.flag_frame=[]
                        else:
                            self.dele_lib_id(self.id_new)
                            self.flag_frame=[]
                else:
                    #if frame - self.save[id_]['frame'] >3:
                    self.flag_frame=[]
                    #self.add_names.append(id_)
                    self.flag_line.append(frame)
                    self.save[id_]['frame']=frame
                    #self.update_frame 
                    if len(self.flag_line)>=3:
                        if np.mean(self.flag_line)==self.flag_line[1]:
                            self.paassag_on(feature,bbox,frame,id_,camera_id)
                            self.flag_line=[]
                        else:
                            self.flag_line=[]
                    return id_
        else:
            self.id_new = self.add_lib(feature,bbox,frame,camera_id,passage=True)
        return self.id_new
    #*                  
    #  *  状态管理函数集   
    #*  
    def hold_on(self,id_):
        self.save[id_]['hold']=True
    def hold_off(self):    
        for id_ in self.save.keys():
            self.save[id_]['hold']=False
        
    def paassag_on(self,feature,bbox,frame,id_,camera_id):
        self.save[id_]['camera'][camera_id]['center']=feature
        self.save[id_]['camera'][camera_id]['bbox']=bbox
        self.save[id_]['lib'].append(feature)
        self.save[id_]['frame']=frame
        self.save[id_]['passage']=True
        #print('%s passage is True'%id_)
        
    def passage_off(self,id_,feature,camera_id):
        self.save[id_]['camera'][camera_id]['center']=feature
        self.save[id_]['passage']=False
        #print('%s passage is False'%id_)
    
    def overlap_on(self,id_,camera_id):
        self.save[id_]['camera'][camera_id]['overlap']=True
        #print('%s overlap is True'%id_)
        
    def overlap_off(self,id_,camera_id):
        self.save[id_]['camera'][camera_id]['overlap']=False
        #print('%s overlap is False'%id_)
        
    def cross_on(self,id_,camera_id):
        self.save[id_]['camera'][camera_id]['cross']=True
        #print('%s cross is True'%id_)
        
    def cross_off(self,id_,camera_id):
        self.save[id_]['camera'][camera_id]['cross']=False
        #print('%s cross is False'%id_)
        
    def update_frame(self,id_,frame):
        self.save[id_]['frame']=frame
        
    def dele_lib_id(self,id_):
        del self.save[id_]
        print('the id %s Remove'%id_)
    
#*                  
#*      检测函数集   
#*                  
    def check_create_id(self,bboxs,im,camera_id):
        list_feature,list_names=[],[]
        for bbox in bboxs:
            (x, y, w, h) = bbox
            (x1, y1, x2, y2)=abs(int(x-w/2)), abs(int(y-h/2)), int(x+w/2), int(y+h/2)
            im_=im[y1:y2,x1:x2]
            im_ = cv2.resize(im_,(64,128))
            list_feature.append(self.compute_feature_cnn(im_))
        for feature in list_feature:
            list_dis,list_overlap_name=[],[]
            for k,v in self.save.items():
                if k!='000' and v['passage']==True and v['hold']==False:
                    #iou1 ,iou2 = self.compute_IOU_myself(bbox,v['bbox'])
                    dis = self.compute_cos(v['camera'][camera_id]['center'],feature)
                    list_dis.append(dis)
                    list_overlap_name.append(k)
            list_names.append(list_overlap_name[list_dis.index(min(list_dis))])
        return list_names

    def check_overlap_id(self,self_id,camera_id):
        over_id=' '
        ids = self.save[self_id]['camera'][camera_id]['overlap_id']
        for id_ in ids:
            if self.save[id_]['hold']==False:
                self.save[id_]['frame']=self.frame
                self.hold_on(id_)
                over_id += ('&%s'%id_) 
            else:
                self.save[self_id]['camera'][camera_id]['overlap_id'].remove(id_)
        if len(self.save[self_id]['camera'][camera_id]['overlap_id'])<2:
            self.dele_lib_id(self_id)
            over_id=' '
        return over_id
    
    def check_frame_id(self,id_,feature,frame):
        list_dis=[]
        for k,v in list(self.save.items()):
            if k==id_:
                if frame-(v['frame'])>=10:
                        if k==id_:
                            for lib_ in v['lib']:
                                list_dis.append(self.compute_cos(lib_,feature))
                            if min(list_dis)<self.dth:
                                return True
                            else:
                                return False
                else:
                    return True
        return False
   #检查bbox是否有交叉
    def check_bboxs_cross(self,bboxs,bbox):    
        ret=False
        (xa, ya, wa, ha)=bbox
        cross_bboxs=copy.copy(bboxs)
        cross_bboxs.remove(bbox)
        for bbox_ in cross_bboxs:
            (xb, yb, wb, hb) = bbox_ 
            distance = self.distEclud(xa,ya,xb,yb)
            if (distance/(wa+wb))>=0.5:
                continue
            if (distance/(wa+wb))<0.5:
                ret=True  
        return ret 

    def check_lib_add(self,featureVector,min_distance_id,bbox,camera_id):
        for k,v in list(self.save.items()):
            list_dis,lib=[],[]
            if  k==min_distance_id:
                n=v['camera'][camera_id]['center']
                v['bbox']=bbox
                #跟新kalman 轨迹预测
                if self.kalman_switch:
                    self.kalman_bbox_update(k,bbox,camera_id)
                if len(v['lib'])>511:
                    for lib_ in v['lib']:
                        list_dis.append(self.compute_cos(lib_,featureVector))
                    num = list_dis.index(min(list_dis))
                    self.save[min_distance_id]['lib'].pop(num)
                    #self.save[min_distance_id]['lib'].pop(0)
                if len(list_dis)>0 and min(list_dis)>self.lib_dth:
                    self.save[min_distance_id]['lib'].append(featureVector)
                    print('\r %s lib len :%d'%(min_distance_id,len(lib)),end=' ')
                    #tqdm.write("lib len %.2f" %(len(lib)/511))
                break
        return int(len(v['lib'])/2)  
    #检查未就绪
    def check_lib_passage(self,featureVector):
        list_id,list_mean=[],[]
        for k,v in list(self.save.items()):
            list_dis=[]
            if k!='000' and v['passage']==False and len(v['lib'])>5:
                lib = v['lib']
                for lib_ in lib:
                    list_dis.append(self.compute_cos(lib_,featureVector))
                list_id.append(k)
                list_mean.append(min(list_dis)) 
        if len(list_mean)>0 and min(list_mean) <self.passage_dth:
            #print(list_mean)
            return list_id[list_mean.index(min(list_mean))] 
        else:
            #print( min(list_mean))
            #if len(list_mean)>0:
                #print(list_mean)
            return ' '      
    
    #*****************
    #
    #   三大模式判断
    #
    #****************
    #*None模式
    def None_mode(self,notebook,im,camera_id):
        CNN_bbox,list_id,list_feature,list_distance,list_center,list_c=[],[],[],[],[],[]
        for bbox,c in notebook.items():
            CNN_bbox.append(bbox)
            (x, y, w, h) = bbox
            #print(bbox)
            (x1, y1, x2, y2)=abs(int(x-w/2)), abs(int(y-h/2)), int(x+w/2), int(y+h/2)
            #print((x1, y1, x2, y2))
            im_=im[y1:y2,x1:x2]
            im_ = cv2.resize(im_,(64,128))
            #featureVector = self.compute_feature(im_)
            featureVector = self.compute_feature_cnn(im_)
            #不交叉不重叠
            distance=0                       
            #所有图片与LIB中passage为TRUE的计算距离
            list_distance_buffer,list_id_buffer,list_center_buffer=[],[],[]
            for k,v in list(self.save.items()):
                if k!='000' and v['passage']==True and v['hold']==False and v['camera'][camera_id]['overlap']==False:# and v['camera'][camera_id]['center']!=[]:
                    n=v['camera'][camera_id]['center']
                    #print(self.save)
                    if self.kalman_switch:
                        kalman_xy=v['camera'][camera_id]['kalman']
                        distance_kalman = self.distEclud(kalman_xy[0],kalman_xy[1],bbox[0],bbox[1])
                        distance_feature = self.compute_cos(n,featureVector)
                        distance =+distance_feature*0.95+distance_kalman*0.01
                    else:
                        distance = self.compute_cos(n,featureVector)
                    list_distance_buffer.append(distance)
                    list_id_buffer.append(k)
                    list_center_buffer.append(n)
            if len(list_distance_buffer)>0 :#and min(list_distance_buffer)<self.reid_dth:
                list_distance.append(list_distance_buffer)
                list_id.append(list_id_buffer)
                list_center.append(list_center_buffer)
                list_feature.append(featureVector)
                list_c.append(c)
            else:
                #ret = self.check_bboxs(bbox_list,bbox) 
                #if ret ==True: 
                id_ = self.add_id(featureVector,bbox,camera_id,self.frame)
                self.lecture[bbox]=[id_,c,0]
        #匈牙利算法
        list_bbox_loss,list_feature_loss,confidences_loss=[],[],[]
        if CNN_bbox!=[] and list_distance!=[]:
                CNN_center,CNN_feature,CNN_id,CNN_c=[],[],[],[]
                #需要id的数量要小于等于就绪id的数量
                if len(CNN_bbox)>len(list_distance[0]):
                    min_distance=[]
                    for id_distanc in list_distance:
                        min_distance.append(min(id_distanc))
                    for i in range((len(CNN_bbox)-len(list_distance[0]))):
                        if len(min_distance)==0:
                            continue
                        num=min_distance.index(max(min_distance))
                        min_distance.pop(num)
                        list_distance.pop(num)
                        list_id.pop(num)
                        list_center.pop(num)
                        list_feature_loss.append(list_feature.pop(num))
                        list_bbox_loss.append(CNN_bbox.pop(num))
                        confidences_loss.append(list_c.pop(num))
                if len(list_distance)>0:
                    np_distance=np.asarray(list_distance)
                    line,column=np_distance.shape[0],np_distance.shape[1]
                    if line!=column:
                        resize_numpy=np.ones((column,column))
                        resize_numpy[:line,:]=np_distance
                        list_Hungary = self.task.Hungary(resize_numpy)
                        for i in range(abs(line-column)):
                            list_Hungary.pop(-1)
                    else:
                        list_Hungary = self.task.Hungary(np_distance)
                    num_CNN=0   
                    print(self.frame,np_distance,list_Hungary,camera_id)
                    for number in list_Hungary:
                        if number<self.reid_dth:
                            min_num = np_distance[num_CNN].flatten().tolist().index(number)
                            #print(min_distance)
                            CNN_id.append(list_id[num_CNN][min_num])
                            CNN_center.append(list_center[num_CNN][min_num])
                            CNN_feature.append(list_feature[num_CNN])
                            CNN_c.append(list_c[num_CNN])
                        else:
                            list_bbox_loss.append(CNN_bbox.pop(num_CNN))
                            list_feature_loss.append(list_feature[num_CNN])
                            confidences_loss.append(list_c[num_CNN])
                        num_CNN+=1
                    
                for bbox,center,feature,id_,c in zip(CNN_bbox,CNN_center,CNN_feature,CNN_id,CNN_c):
                     #检查除id_me外是否与数据库里的bbox重叠
                    center_ = (center*0.5)+(feature*0.5)
                    for name in self.camera_ids:
                        self.save[id_]['camera'][name]['center'] =center_
                    self.save[id_]['frame'] =self.frame
                    #print(bbox[2]*bbox[3],self.frame)
                    color_=self.check_lib_add(feature,id_,bbox,camera_id)
                    self.lecture[bbox]=[id_,c,color_]
                    self.hold_on(id_)
                    self.cross_off(id_,camera_id)

                #没有ID,重新add_id    
                #print(list_bbox_loss,list_feature_loss,confidences)
                for bbox,featureVector,c in zip(list_bbox_loss,list_feature_loss,confidences_loss):
                    #print(bbox,featureVector,c)
                    id_ = self.add_id(featureVector,bbox,camera_id,self.frame)
                    self.lecture[bbox]=[id_,c,0]
    def Cross_mode(self,notebook,im,camera_id):     
        CNN_bbox,CNN_C,list_id,list_feature,list_distance,list_center,list_c=[],[],[],[],[],[],[]
        notebook_to_None={}
        for bbox,c in notebook.items():
            CNN_bbox.append(bbox)
            (x, y, w, h) = bbox
            #print(bbox)
            (x1, y1, x2, y2)=abs(int(x-w/2)), abs(int(y-h/2)), int(x+w/2), int(y+h/2)
            #print((x1, y1, x2, y2))
            im_=im[y1:y2,x1:x2]
            im_ = cv2.resize(im_,(64,128))
            #featureVector = self.compute_feature(im_)
            featureVector = self.compute_feature_cnn(im_)
            #不交叉不重叠
            distance=0
            #所有图片与LIB中passage为TRUE的计算距离
            list_distance_buffer,list_id_buffer,list_center_buffer=[],[],[]
            for k,v in list(self.save.items()):
                if k!='000' and v['passage']==True and v['hold']==False and v['camera'][camera_id]['overlap']==False and v['camera'][camera_id]['center']!=[]:
                    n=v['camera'][camera_id]['center']
                    if self.kalman_switch:
                        kalman_xy=v['camera'][camera_id]['kalman']
                        distance_kalman = self.distEclud(kalman_xy[0],kalman_xy[1],bbox[0],bbox[1])
                        distance_feature = self.compute_cos(n,featureVector)
                        distance =+distance_feature*0.95+distance_kalman*0.01
                    else:
                        distance = self.compute_cos(n,featureVector)
                    list_distance_buffer.append(distance)
                    list_id_buffer.append(k)
                    list_center_buffer.append(n)
            if len(list_distance_buffer)>0 :#and min(list_distance_buffer)<self.reid_dth:
                list_distance.append(list_distance_buffer)
                list_id.append(list_id_buffer)
                list_center.append(list_center_buffer)
                list_feature.append(featureVector)
                list_c.append(c)
            else:
                notebook_to_None[bbox]=c
        #匈牙利算法
        list_bbox_loss,confidences_loss=[],[]
        if CNN_bbox!=[] and list_distance!=[]:
                CNN_center,CNN_feature,CNN_id,CNN_i,CNN_c=[],[],[],[],[]
                #需要id的数量要小于等于就绪id的数量
                if len(CNN_bbox)>len(list_distance[0]):
                    min_distance=[]
                    for id_distanc in list_distance:
                        min_distance.append(min(id_distanc))
                    for i in range((len(CNN_bbox)-len(list_distance[0]))):
                        if len(min_distance)==0:
                            continue
                        num=min_distance.index(max(min_distance))
                        min_distance.pop(num)
                        list_distance.pop(num)
                        list_id.pop(num)
                        list_center.pop(num)
                        list_feature.pop(num)
                        list_bbox_loss.append(CNN_bbox.pop(num))
                        confidences_loss.append(list_c.pop(num))
                if len(list_distance)>0:
                    np_distance=np.asarray(list_distance)
                    line,column=np_distance.shape[0],np_distance.shape[1]
                    if line!=column:
                        resize_numpy=np.ones((column,column))
                        resize_numpy[:line,:]=np_distance
                        list_Hungary = self.task.Hungary(resize_numpy)
                        for i in range(abs(line-column)):
                            list_Hungary.pop(-1)
                    else:
                        list_Hungary = self.task.Hungary(np_distance)
                    num_CNN=0
                    for number in list_Hungary:
                        if number<self.reid_dth:
                            min_num = np_distance[num_CNN].flatten().tolist().index(number)
                            CNN_id.append(list_id[num_CNN][min_num])
                            CNN_center.append(list_center[num_CNN][min_num])
                            CNN_feature.append(list_feature[num_CNN])
                            CNN_c.append(list_c[num_CNN])
                        else:
                            list_bbox_loss.append(CNN_bbox.pop(num_CNN))
                            confidences_loss.append(list_c[num_CNN])
                        num_CNN+=1
                for bbox,center,feature,id_,c in zip(CNN_bbox,CNN_center,CNN_feature,CNN_id,CNN_c):
                     #检查除id_me外是否与数据库里的bbox重叠
                    center_ = (center*0.5)+(feature*0.5)
                    for name in self.camera_ids:
                        self.save[id_]['camera'][name]['center'] =center_
                    self.save[id_]['frame'] =self.frame
                    #print(bbox[2]*bbox[3],self.frame)
                    color_=self.check_lib_add(feature,id_,bbox,camera_id)
                    self.lecture[bbox]=[id_,c,color_]
                    self.hold_on(id_)
                    self.cross_on(id_,camera_id)
                    #self.overlap_off(id_)
                #没有ID,送入None判断继续进行reid 
                for bbox,c in zip(list_bbox_loss,confidences_loss):
                    notebook_to_None[bbox]=c
        self.None_mode(notebook_to_None,im,camera_id)
    #OverLap 模式
    def Overlap_mode(self,notebook,im,camera_id):
        my_hold_ids=[]
        for root_bbox,v in notebook.items():
            over_id=''
            (x, y, w, h) = root_bbox
            (x1, y1, x2, y2)=abs(int(x-w/2)), abs(int(y-h/2)), int(x+w/2), int(y+h/2)
            im_=im[y1:y2,x1:x2]
            im_ = cv2.resize(im_,(64,128))
            featureVector = self.compute_feature_cnn(im_)
            
            if v[0]=='create':
                c=v[1]
                create_bboxs=v[2]
                if len(create_bboxs)>=2:
                    list_ids = self.check_create_id(create_bboxs,im,camera_id)
                    if len(list_ids)>=len(create_bboxs):
                        overlap_id = self.add_lib_overlab(featureVector,root_bbox,self.frame,list_ids,camera_id)
                        #print(list_ids)
                        my_hold_ids.append(overlap_id)
                        self.lecture[root_bbox]=[over_id,c,0]
                    else:
                        for id_ in list_ids:
                            over_id+=id_
                        self.lecture[root_bbox]=[over_id,c,0]
                else:
                    print('错误构建')
                    self.lecture[root_bbox]=[' ',c,0]
                print(self.frame,'create id %s'%over_id)
                
            if v[0]=='inherit':
                c,self_id=v[1],v[2]
                over_id = self.check_overlap_id(self_id,camera_id)
                if over_id!=' ':
                    color_ = self.check_lib_add(featureVector,self_id,root_bbox,camera_id)
                    self.save[self_id]['frame']=self.frame
                    self.save[self_id]['camera'][camera_id]['overlap']=True
                    self.save[self_id]['camera'][camera_id]['center']=featureVector
                    my_hold_ids.append(self_id)
                    self.lecture[root_bbox]=[over_id,c,color_]
                    print(self.frame,'inherit id %s'%over_id)
            if v[0]=='combine_split':
                c,bboxs,self_id=v[1],v[2],v[3]
                list_features,list_distances=[],[]
                feture_source=self.save[self_id]['camera'][camera_id]['center']
                #去除与self_id相同的bbox 
                for bbox_ in bboxs:
                    (x, y, w, h) = bbox_
                    (x1, y1, x2, y2)=abs(int(x-w/2)), abs(int(y-h/2)), int(x+w/2), int(y+h/2)
                    im_=im[y1:y2,x1:x2]
                    im_ = cv2.resize(im_,(64,128))
                    f_ = self.compute_feature_cnn(im_)
                    list_features.append(f_)
                    list_distances.append(self.compute_cos(f_,feture_source))
                #if len(list_distances)>0:
                list_features.pop(list_distances.index(min(list_distances)))
                bboxs.pop(list_distances.index(min(list_distances)))
                #检查bbox的id 是否在self_id 不在为合并
                #print('combin have:%d'%len(bboxs))
                for bbox_,feature1 in zip(bboxs,list_features):
                    min_diantance,min_feature,min_bbox=[],[],[]
                    for id_ in self.save[self_id]['camera'][camera_id]['overlap_id']:
                        list_=[]
                        for feature2 in self.save[id_]['lib']:
                        #feature2 = self.save[id_]['center']
                            list_.append(self.compute_cos(feature1,feature2))
                        min_diantance.append(min(list_))
                    #发现新的ID上车
                    if min(min_diantance)>self.reid_dth:
                        #找到bbox匹配的id
                        min_distance,min_id=[],[]
                        #feature 找寻id
                        for k_,v_ in self.save.items():
                            if k_!='000' and v_['passage']==True and v_['hold']==False:
                                center = self.save[k_]['camera'][camera_id]['center']
                                min_distance.append(self.compute_cos(feature1,center))
                                min_id.append(k_)
                        #外来合并的判断
                        #继承一个id
                        #有可用ID
                        if len(min_distance)>0:
                            #使用最像的ID
                            id_combine = min_id[min_distance.index(min(min_distance))]
                            #添加id到overlap_id 里
                            if id_combine not in self.save[self_id]['camera'][camera_id]['overlap_id']:
                                if len(self.save[id_combine]['camera'][camera_id]['overlap_id'])>0:
                                    self.save[self_id]['camera'][camera_id]['overlap_id']+=self.save[id_combine]['camera'][camera_id]['overlap_id']
                                    self.save[self_id]['camera'][camera_id]['overlap_id']=list(set(self.save[self_id]['camera'][camera_id]['overlap_id']))
                                    self.dele_lib_id(id_combine)
                                else:
                                    self.save[self_id]['camera'][camera_id]['overlap_id'].append(id_combine)
                            #取出ID合集
                            over_id = self.check_overlap_id(self_id,camera_id)
                            if over_id!=' ':
                                color_ = self.check_lib_add(featureVector,self_id,root_bbox,camera_id)
                                #print(root_bbox)
                                self.save[self_id]['frame']=self.frame
                                self.save[self_id]['camera'][camera_id]['center']=featureVector
                                self.save[self_id]['camera'][camera_id]['overlap']=True
                                my_hold_ids.append(self_id)
                                self.lecture[root_bbox]=[over_id,c,color_]
                            #my_lecture[self_id]=[root_bbox,over_id,c,color_]
                            print(self.frame,'combine id %s'%over_id)
                        #外面都找不到，不做处理
                        else:
                            #取出ID合集
                            over_id = self.check_overlap_id(self_id,camera_id)
                            if over_id!=' ':
                                color_ = self.check_lib_add(featureVector,self_id,root_bbox,camera_id)
                                self.save[self_id]['frame']=self.frame
                                self.save[self_id]['camera'][camera_id]['center']=featureVector
                                self.save[self_id]['camera'][camera_id]['overlap']=True
                                my_hold_ids.append(self_id)
                                self.lecture[root_bbox]=[over_id,c,color_]
                            #my_lecture[self_id]=[root_bbox,over_id,c,color_]
                            #print(self.frame,'no find id %s'%over_id)
                    #已在车上
                    else:
                        #取出ID合集
                        over_id = self.check_overlap_id(self_id,camera_id)
                        if over_id!=' ':
                            color_ = self.check_lib_add(featureVector,self_id,root_bbox,camera_id)
                            self.save[self_id]['frame']=self.frame
                            self.save[self_id]['camera'][camera_id]['center']=featureVector
                            self.save[self_id]['camera'][camera_id]['overlap']=True
                            my_hold_ids.append(self_id)
                            self.lecture[root_bbox]=[over_id,c,color_]
                        #my_lecture[self_id]=[root_bbox,over_id,c,color_]
                        #print(self.frame,'find id existing %s'%over_id)
            for id_ in my_hold_ids:
                if id_ in self.save.keys():
                    self.hold_on(id_)
                
    def deal_cross(self,bboxs,confidences,iou_button):
        notebook_deal={}
        overlap_del_bbox,select_bbox=[],[]
        myself_bboxs=copy.copy(bboxs)
        myself_confidences=copy.copy(confidences)
        if len(bboxs)>=2:
            for i in range(len(bboxs)):
                for j in range(i+1,len(bboxs)):
                        iou1,iou2 = self.compute_IOU_myself(bboxs[i],bboxs[j])
                        if iou1 >iou_button or iou2>iou_button:
                            select_bbox.append(bboxs[i])
                            select_bbox.append(bboxs[j])
        select_bbox = list(set(select_bbox))
        if len(select_bbox)>=2:
            create_bbox=[]
            #找到重叠的bbox,用来生成新的overlap ID
            for i in range(len(select_bbox)):
                for j in range(i+1,len(select_bbox)):
                    iou1,iou2 = self.compute_IOU_myself(select_bbox[i],select_bbox[j])
                    if iou1 >iou_button or iou2>iou_button: 
                        #print(select_bbox[i],select_bbox[j])
                        bbox_combine = self.combine_bbox(select_bbox[i],select_bbox[j])
                        create_bbox.append(bbox_combine)
                        notebook_deal[bbox_combine]=[select_bbox[i],select_bbox[j]]
                        #print('合成的：',bbox_combine)
                        overlap_del_bbox.append(select_bbox[i])
                        overlap_del_bbox.append(select_bbox[j])
            #剔出所有被合并的bbox
            #print('sourece:',myself_bboxs)
            for bbox_ in overlap_del_bbox:
                if bbox_ in myself_bboxs:
                    myself_confidences.pop(myself_bboxs.index(bbox_))
                    myself_bboxs.remove(bbox_)
            #新生的bbox用于创建ID
            for bbox_create in create_bbox:
                myself_bboxs.append(bbox_create)
                myself_confidences.append(0)
            #print(myself_bboxs,myself_confidences)
        return myself_bboxs,myself_confidences,overlap_del_bbox,notebook_deal
    #overlap bbox查找函数
    def check_bboxs_overlap(self,bboxs,confidences,im,camera_id):
        myself_bboxs_,myself_confidences_,overlap_del_bbox_,notebook_ = self.deal_cross(copy.copy(bboxs),copy.copy(confidences),0.6)
        bbox_root=copy.copy(myself_bboxs_)
        #print(myself_bboxs_,myself_confidences_)
        myself_bboxs,myself_confidences,overlap_del_bbox,notebook_ = self.deal_cross(copy.copy(myself_bboxs_),copy.copy(myself_confidences_),0.2)
        out_bboxs=copy.copy(myself_bboxs)
        #print(myself_bboxs,myself_confidences)
        #过滤完成
        notebook_overlap={}
        #print(,list_new_bbox)
        #合并 / 分裂
        if len(notebook_.keys())>0:
            #print(notebook_)
            #如果有overlapbbox，找到合并的bbox的父类bboxs，和需要合并的id
            for bbox_new,bbox_father in notebook_.items():
                notebook_combine={}
                for k,v in self.save.items():
                    if k!='000' and v['camera'][camera_id]['overlap']==True:
                        iou1,iou2 = self.compute_IOU_myself(v['camera'][camera_id]['bbox'],bbox_new)
                        if iou1>0.2 or iou2>0.2:
                            notebook_combine[bbox_new]=[bbox_father,k]
                #合并/分裂发生：        
                if len(notebook_combine.keys())>0:
                     for bbox_,v_ in notebook_combine.items():
                        num = myself_bboxs.index(bbox_)
                        notebook_overlap[myself_bboxs.pop(num)]=['combine_split',myself_confidences.pop(num),v_[0],v_[1]]
                        #占用id 不继承
                        self.hold_on(v_[1])
                #新生发生
                else:
                    #print('bbox_father: ',bbox_new,bbox_father)
                    num = myself_bboxs.index(bbox_new)
                    notebook_overlap[myself_bboxs.pop(num)]=['create',myself_confidences.pop(num),bbox_father]
        #所有没被占用的id 找到继承的bbox,找不到删除
        if len(myself_bboxs)!=0:
            for k,v in self.save.items():
                if k!='000' and v['camera'][camera_id]['overlap']==True and v['hold']==False:
                    list_bboxs,list_ious,list_c=[],[],[]
                    for bbox_,c_ in zip(myself_bboxs,myself_confidences):
                        iou1,iou2 = self.compute_IOU_myself(v['camera'][camera_id]['bbox'],bbox_)
                        if iou1>0.2 or iou2>0.2:
                            list_bboxs.append(bbox_)
                            list_c.append(c_)
                            list_ious.append(iou1)
                    #继承
                    if len(list_bboxs)>0:
                        num = list_ious.index(max(list_ious))
                        bbox_inheit=list_bboxs[num]
                        del_num = myself_bboxs.index(bbox_inheit)
                        notebook_overlap[bbox_inheit]=['inherit',list_c[num],k]
                        myself_bboxs.pop(del_num)
                        myself_confidences.pop(del_num)
                        #self.hold_on(k)
        #print(notebook_overlap)
        #print(self.save['000'])
        return notebook_overlap,myself_bboxs,out_bboxs,myself_confidences
    #bbox选则函数
    def check_bboxs_select(self,bboxs,confidences,im,camera_id):
        #3个状态存储
        cross_notebook,none_notebook,overlap_notebook={},{},{}
        #1.从bboxs中找到overlap继承的bbox,并剔出
        #2.从bboxs中找到合并的，并剔出
        overlap_notebook,bbox_none,out_bboxs,confidences_ = self.check_bboxs_overlap(bboxs,confidences,im,camera_id)
        if len(bbox_none)>0:
            for bbox_,c in zip(bbox_none,confidences_):
                ret = self.check_bboxs_cross(bbox_none,bbox_)
                if ret ==True:
                    cross_notebook[bbox_]=c
                else:
                    none_notebook[bbox_]=c          
        return none_notebook,cross_notebook,overlap_notebook,out_bboxs
    #主函数 
    def cross_video(self,list_bbox_all,list_conf_all,list_im): 
        #维护LIB
        self.manage_lib(self.frame)
        out_bboxs,color,confidence,identities=[],[],[],[]
        for bbox_list,confidences,im,camera_id in zip(list_bbox_all,list_conf_all,list_im,self.camera_ids):
            out_bboxs_,color_,confidence_,identities_=[],[],[],[]
            self.lecture={}
            if bbox_list!=[]:
                #bbox选择分类处理
                none_notebook,cross_notebook,overlap_notebook,out_bboxs_ = self.check_bboxs_select(bbox_list,confidences,im,camera_id)

                self.None_mode(none_notebook,im,camera_id)
                self.Cross_mode(cross_notebook,im,camera_id)
                self.Overlap_mode(overlap_notebook,im,camera_id)
                #读取识别的结果id c color           
                for bbox in out_bboxs_:
                    if bbox in self.lecture.keys():
                        identities_.append(self.lecture[bbox][0])
                        confidence_.append(self.lecture[bbox][1])
                        color_.append(self.lecture[bbox][2])
                out_bboxs.append(out_bboxs_)
                confidence.append(confidence_)
                color.append(color_)
                identities.append(identities_)
                #print(self.frame,out_bboxs,self.lecture) 
                self.hold_off()
                self.frame+=1
            else:
                out_bboxs.append(out_bboxs_)
                confidence.append(confidence_)
                color.append(color_)
                identities.append(identities_)
        return out_bboxs,identities,confidence,color,self.frame
