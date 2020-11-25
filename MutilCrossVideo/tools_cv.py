import cv2
import os
import numpy as np
import random
import copy
import time
import glob
from IPython.display import clear_output

def ImRateResize(im,rate):
    return cv2.resize(im,(int(im.shape[1]*rate),int(im.shape[0]*rate)))

def ImMask(im,points):
    im0 = np.zeros((im.shape)).astype(np.uint8)
    im0 = cv2.fillConvexPoly(im0, points.astype(np.int), (1,1,1), cv2.LINE_AA)
    return im*im0

def ImRoi(im,point_list):
    x_min,x_max = min(point_list[:,0]),max(point_list[:,0])
    y_min,y_max = min(point_list[:,1]),max(point_list[:,1])
    point_list[:,0],point_list[:,1] = point_list[:,0]-x_min,point_list[:,1]-y_min
    x_min,x_max,y_min,y_max = int(x_min),int(x_max),int(y_min),int(y_max)
    im = im[y_min:y_max,x_min:x_max]
    im = ImMask(im,point_list)
    return im

def ImRoi(im,point_list_):
    point_list = point_list_.copy()
    x_min,x_max = min(point_list[:,0]),max(point_list[:,0])
    y_min,y_max = min(point_list[:,1]),max(point_list[:,1])
    point_list[:,0],point_list[:,1] = point_list[:,0]-x_min,point_list[:,1]-y_min
    x_min,x_max,y_min,y_max = int(x_min),int(x_max),int(y_min),int(y_max)
    im = im[y_min:y_max,x_min:x_max]
    im = ImMask(im,point_list)
    return im

def ImConcat(ims,fmt='w'):
    '''拼接多张不同大小的图片 exp:ims=[im0,im1...]'''
    w,h = 0,0
    for im in ims:
        h1,w1 = im.shape[0:2]
        if w<w1:
            w=w1
        if h<h1:
            h=h1
    b = np.zeros((h,w,3)).astype(np.uint8)
    c_ims = []
    for im in ims:
        b1 = b.copy()
        h1,w1 = im.shape[0:2]
        b1[0:h1,0:w1] = im
        c_ims.append(b1)
    if fmt=='h':
        return np.vstack(c_ims)
    if fmt=='w':
        return np.hstack(c_ims)

def TwoPointToRectangle(p0,p1,rate=3):
    '''由两点生成一个矩形的四个点'''
    [x0,y0],[x1,y1] = p0,p1
    l  = np.linalg.norm(p0-p1)/rate
    k  = (y1-y0+1e-7)/(x1-x0+1e-7)
    b  = y1-k*x1
    l3,l2 = [-1/k, y0+(x0/k)],[-1/k, y1+(x1/k)]
    l1,l0 = [k, -l*np.sqrt(k**2+1)+b],[k, l*np.sqrt(k**2+1)+b]
    x03 = -(l0[1]-l3[1])/(l0[0]-l3[0]+1e-7)
    y03 = l0[0]*x03+l0[1]
    x02 = -(l0[1]-l2[1])/(l0[0]-l2[0]+1e-7)
    y02 = l0[0]*x02+l0[1]
    x12 = -(l1[1]-l2[1])/(l1[0]-l2[0]+1e-7)
    y12 = l1[0]*x12+l1[1]
    x13 = -(l1[1]-l3[1])/(l1[0]-l3[0]+1e-7)
    y13 = l1[0]*x13+l1[1]
    return np.array([[x03,y03],[x02,y02],[x12,y12],[x13,y13]]).astype(np.int)

red,green = (0,0,255),(0,255,0)
txt_font = cv2.FONT_HERSHEY_SIMPLEX
fourcc = cv2.VideoWriter_fourcc(*'XVID')

SkeletonFmt = {'vp17'  :[[3,2],[2,1],[1,0],[0,4],[4,5],[5,6],[16,15],[15,14],[14,8],[8,11],[11,12],[12,13],[0,7],[7,8],[8,9],[9,10]],
                'smpl24':[[23,21],[21,19],[19,17],[17,14],[14,9],[9,13],[13,16],[16,18],[18,20],[20,22],[15,12],[12,9],[9,6],[6,3],[3,0],[0,1],[1,4],[4,7],[7,10],[0,2],[2,5],[5,8],[8,11],[0,6],[6,12]],
                'op25'  :[[17,15],[15,0],[0,16],[16,18],[4,3],[3,2],[2,1],[1,5],[5,6],[6,7],[0,1],[1,8],[23,22],[22,11],[11,10],[10,9],[9,8],[8,12],[12,13],[13,14],[14,19],[19,20],[14,21]],
                'op25b' :[[10,8],[8,6],[6,17],[17,5],[5,7],[7,9],[4,2],[2,0],[0,1],[1,3],[18,0],[0,17],[17,12],[12,14],[14,16],[16,22],[22,23],[16,24],[17,11],[11,13],[13,15],[15,19],[19,20],[15,21]],
                'coco17':[[0,1],[1,3],[0,2],[2,4],[10,8],[8,6],[6,5],[5,7],[7,9],[16,14],[14,12],[12,11],[11,13],[13,15],[6,12],[5,11]],
                'mpii16':[[10,11],[11,12],[12,7],[7,13],[13,14],[14,15],[9,8],[8,7],[7,6],[6,3],[3,4],[4,5],[6,2],[2,1],[1,0]],
                'kn32'  :[[31,30],[30,27],[27,28],[28,29],[27,26],[26,3],[3,2],[2,1],[1,0],[0,18],[18,19],[19,20],[20,21],[0,22],[22,23],[23,24],[24,25],[16,15],[15,14],[14,17],[14,13],[13,12],[12,11],[11,2],[2,4],[4,5],[5,6],[6,7],[7,8],[8,9],[7,10]]}

def DrawBboxs(im, bboxes, bboxes_index=None, im_index=None, color=None):
    if color is None:
        color=(0,0,255)
    if im_index is not None:
        cv2.putText(im, f'{im_index}', (10,30), txt_font, 0.6, color, 2,cv2.LINE_AA)
    
    N = bboxes.shape[0]
    for i in range(N):
        bbox = bboxes[i].astype(np.int)
        p0,p1 = (bbox[0],bbox[1]),(bbox[2],bbox[3])
        if bboxes_index is not None:
            cv2.putText(im, f'{bboxes_index[i]}', p0, txt_font, 1, color, 2,cv2.LINE_AA)
        cv2.rectangle(im,p0,p1,color,1,cv2.LINE_AA)
    return im

def IsLine(im, p0, p1, color=(0,0,0), size=1):
    if (p0[0]!=0 or p0[1]!=0)and(p1[0]!=0 or p1[1]!=0):
        return cv2.line(im,p0,p1,color,size,cv2.LINE_AA)
    return im

def DrawKps(im, kp2ds, fmt='op25b', color=None, show_point=0, show_point_txt=0, show_lines=1):
    color_point,color_line = (random.randint(0,255),random.randint(0,255),random.randint(0,255)),(random.randint(0,255),random.randint(0,255),random.randint(0,255))
    if color is not None:
        color_point,color_line = color,color
    ### kp2ds shape is NxMxC N是当前帧的人数,M是骨骼点数,C一般为2
    N,M,C = kp2ds.shape
    color2ds = kp2ds.copy().astype(np.int)[:,:,0:2]
    if C==3:
        cfds = kp2ds.copy()[:,:,-1]
        cfds = np.around(cfds,decimals=2)
    for i in range(N):
        points = []
        for j in range(M):
            p = tuple(color2ds[i,j])
            points.append(p)
            if show_point is not 0:
                im = cv2.circle(im, p, show_point, color_point, -1, cv2.LINE_AA)
            if show_point_txt is not 0:
                im = cv2.putText(im,f'{j}', p, txt_font, show_point_txt, color_point, 1, cv2.LINE_AA)
            if C==3:
                im = cv2.putText(im,f'{cfds[i,j]}', p, txt_font, 0.5, color_point, 1, cv2.LINE_AA)
        if show_lines is not 0:
            for skeleton in SkeletonFmt[fmt]:
                IsLine(im, points[skeleton[0]], points[skeleton[1]], color_line, show_lines)
    return im

def VideoShowKp2ds1(kp2ds, bg='', fps=1, fmt='vp17', color=None, save=None, window='A', show_point=0, show_point_txt=0, show_lines=2):
    '''kp2ds shape:NxMxC'''
    intval = int(1000/fps)
    [N,M,C] = kp2ds.shape
    if os.path.isfile(bg) is True:
        cap = cv2.VideoCapture(bg)
        w,h,fps,max_num = int(cap.get(3)),int(cap.get(4)),int(cap.get(5)),int(cap.get(7))
        if save is not None:
            save_video = cv2.VideoWriter(save, fourcc, fps, (w,h))
        for i in range(min(N,max_num)):
            re,im = cap.read()
            DrawKps(im, kp2ds[i:i+1,:], fmt, color, show_point, show_point_txt, show_lines)
            cv2.putText(im,f'{i}', (30,30), txt_font, 1, (color), 1, cv2.LINE_AA)
            if save is not None:
                save_video.write(im)
            cv2.imshow(window, im)
            if cv2.waitKey(intval)&0xff==ord('q'):
                break
        cap.release()
        if save is not None:
            save_video.release()
    if bg is '':
        w,h = 1920,1080
        im_bk = np.zeros((h,w,3))+255
        for i in range(N):
            im = DrawKps(im_bk.copy(), kp2ds[i:i+1,:], fmt, color, show_point, show_point_txt, show_lines)
            cv2.imshow(window, im)
            if cv2.waitKey(intval)&0xff==ord('q'):
                break
    cv2.destroyAllWindows()

def show_bboxes(bboxes, bg='', color=(255,0,0), fps=30, save=None, mask=False, size=False):
    '''bboxes shape Nx4  save is path  size=1.2'''
    intval = int(1000/fps)
    cap = cv2.VideoCapture(bg)
    max_index = int(cap.get(7))
    w,h = int(cap.get(3)),int(cap.get(4))
    if save is not None:
        save_cap = cv2.VideoWriter(save, cv2.VideoWriter_fourcc(*'XVID'), int(cap.get(5)), (w,h))
    if size is not False:
        bboxes[:,[0,1]]-=size[0]
        bboxes[:,[2,3]]+=size[1]
        bboxes = bboxes.astype(np.int)
        bboxes[:,2] = np.where(bboxes[:,2]>=w,w-1,bboxes[:,2])
        bboxes[:,3] = np.where(bboxes[:,3]>=h,h-1,bboxes[:,3])
    for i in range(max_index):
        re,im = cap.read()
        if re is False:
            break
        if mask:
            pmin,pmax = bboxes[i,0:2],bboxes[i,2:4]
            im_mask = np.zeros((h,w,3)).astype(np.uint8)
            im_mask[pmin[1]:pmax[1],pmin[0]:pmax[0]] = im[pmin[1]:pmax[1],pmin[0]:pmax[0]]
            im = im_mask
        else:
            im = DrawBboxs(im, bboxes[i:i+1], i, color)
        cv2.imshow('A', im)
        save_cap.write(im)
        if cv2.waitKey(intval)&0xff==ord('q'):
            break
    cap.release()
    if save is not None:
        save_cap.release()
    cv2.destroyAllWindows()
    

def to_bbox_xcycwh(bbox):
    bbox1 = bbox.copy()
    bbox[:,0] = (bbox1[:,0]+bbox1[:,2])/2
    bbox[:,1] = (bbox1[:,1]+bbox1[:,3])/2
    bbox[:,2] -= bbox1[:,0]
    bbox[:,3] -= bbox1[:,1]
    return bbox

def fuzzy_im(im,bboxs):
    im_result = cv2.blur(copy.deepcopy(im),(30,30))
    for bbox in bboxs:
        bbox = bbox.astype(np.int)
        im_result[bbox[1]:bbox[3],bbox[0]:bbox[2]] = im[bbox[1]:bbox[3],bbox[0]:bbox[2]]
    return im_result

def processing_video_to_im(video_fpath, im_format='jpg', inteval=60, save=''):
    folder = video_fpath[:-4]
    if save is not '':
        folder = save
    if os.path.isdir(folder) is False:
        os.makedirs(folder)
    inteval_index = 0
    cap = cv2.VideoCapture(video_fpath)
    index = 0
    t = time.time()
    while True:
        ret,im = cap.read()
        if inteval_index != inteval:
            index += 1
            inteval_index += 1
            continue
        inteval_index = 0
        if ret is False:
            break
        i = '%06d'%index
        print('\r processing: %s  time: %f' % (str(i),time.time()-t), end='')
        cv2.imwrite(f'{folder}/{i}.{im_format}',im)
        index+=1
    cap.release()

def rectangle_to_square(im):
    ori_h,ori_w,_ = im.shape
    if ori_h<ori_w:
        axis=0
    else:
        axis=1
    new_max,new_min = max(ori_h,ori_w),min(ori_h,ori_w)
    add = [new_min]*(new_max-new_min)
    new_im = np.insert(im, add, values=0, axis=axis)
    return new_im



def im_rotate(im,center,degree):
    h,w,_ = im.shape
    M = cv2.getRotationMatrix2D(center, degree, 1.0)
    im_rotate = cv2.warpAffine(im, M, (w, h))
    return im_rotate

def im_to_video(fpath, fmt='jpg', fps=30, save=''):
    save_video_fpath = f'{os.path.dirname(fpath)}_save.mp4'
    if save is not '':
        save_video_fpath = save
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    im_shape = cv2.imread(glob.glob(f'{fpath}/*.{fmt}')[0]).shape
    out_video = cv2.VideoWriter(save_video_fpath,fourcc,fps,(im_shape[1],im_shape[0]))
    for im_fpath in glob.glob(f'{fpath}/*.{fmt}'):
        print('\r %s' % (os.path.basename(im_fpath)), end='')
        out_video.write(cv2.imread(im_fpath))
    out_video.release()
    
def im_rate_resize(im,rate):
    return cv2.resize(im,(int(im.shape[1]*rate),int(im.shape[0]*rate)))

def video_sync_player(video_list=[], save_fpath='', video_txt=[], size=0.5):
    # video_list: len is 0,1,2,3
    cap_list = []
    for cap_fpath in video_list:
        cap_list.append(cv2.VideoCapture(cap_fpath))
    index = 0
    while True:
        im_list = []
        im_index = 0
        for cap in cap_list:
            ret,im = cap.read()
            if ret is False:
                return 0
            im = cv2.putText(im,f'{video_txt[im_index]}',(50,50),txt_font,1.4,(0,0,255),2,cv2.LINE_AA)
            im_index+=1
            im_list.append(im)
        h,w=im_list[0].shape[0:2]
        re = np.zeros((h*2,w*2,3)).astype(np.uint8)
        if len(im_list)==2:
            re[0:h,0:w],re[0:h,w:2*w] = im_list[0],im_list[1]
        if len(im_list)==3:
            re[0:h,0:w],re[0:h,w:2*w],re[h:2*h,0:w] = im_list[0],im_list[1],im_list[2]
        if len(im_list)==4:
            re[0:h,0:w],re[0:h,w:2*w],re[h:2*h,0:w],re[h:2*h,w:2*w] = im_list[0],im_list[1],im_list[2],im_list[3]
        im_index = '%06d'%index
        if save_fpath is not '':
            cv2.imwrite(f'{save_fpath}/{im_index}.jpg',re)
        print('\r index: %s.jpg'%(im_index),end='')
        index+=1
        cv2.imshow('A', im_rate_resize(re,size))
        if cv2.waitKey(1)&0xff==ord('q'):
            break
    cv2.destroyAllWindows()
        
def cv2_hyaline_line(im, p0, p1, color=(255,0,0), thickness=10, visibility=0.7):
    im0 = np.ones((im.shape)).astype(np.uint8)+255
    cv2.line(im0, p0, p1, color, thickness, cv2.LINE_AA)
    dst = cv2.addWeighted(im,1,im0,visibility,0)
    return dst

def cv2_hyaline_text(im, text, postion, color=(255,0,0), fontscale=3, thickness=3, visibility=0.7):
    im0 = np.ones((im.shape)).astype(np.uint8)+255
    cv2.putText(im0, text, postion, cv2.FONT_ITALIC, fontscale, color, thickness, 20)
    dst = cv2.addWeighted(im,1,im0,visibility,0)
    return dst

def cv2_hyaline_circle(im, postion, color=(255,0,0), fontscale=3, thickness=3, visibility=0.7):
    im0 = np.ones((im.shape)).astype(np.uint8)+255
    cv2.circle(im,postion,fontscale,color,thickness)
    dst = cv2.addWeighted(im,1,im0,visibility,0)
    return dst