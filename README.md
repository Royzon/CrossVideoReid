# CrossVideoReid

## PCA版本  
匈牙利算法  
kalman 轨迹预测  

## CNN 版本
### 2020/7/13
双人版本 2020-07-13_V5_REID_CNN_.ipynb  
解决问题：  
1.交叉出现新id的情况  
2.重叠超时Reid的问题  
3.交叉碰撞问题  
4.重叠碰撞问题  
解决办法：  
增加 在交叉的情况下，匈牙利算法包含对新出现的bbox的处理  
增加 kalman预测bbox不会离开遮挡者bbox，利用IOU判断，预测的bbox与其他bbox 无交叉停止kalman跟踪  
![image](https://github.com/MakerCollider/20200623-haipeng_reid/blob/master/2020.07.01%2000_00_00-00_00_30.gif)  

### 2020/7/16
双人无Kalman跟踪版本 2020-07-16_V5_REID_CNN_Nokalman.ipynb  
解决问题：
因为多人情况复杂，kalman跟踪系统不能很好的表示所有可能
so，使用不同的存储系统只利用feature的不同区分出ID跟关键
分情况比较：交叉ID 重叠ID 无状态ID，在状态改变时，利用记录的bbox及父子关系，判断状态改变的ID，主要记录交叉的时候与重叠的时候的差别  
针对Kalman轨迹跟踪不准确且情况多变的情况进行修改：  
具体操作：  
增加状态：cross_from_overlap  
出现重叠查找最近的两个bbox的ID并记录，产生新的重叠ID ,保持追踪。  
被记录的ID一直处于占用状态 ，保持确认状态 ，直到重叠ID消失释放  
重叠ID消失的判断：出现交叉时候的bbox 只要有一个与 重叠ID的bbox重叠即判断为重叠消失  
重叠消失使用限定匈牙利算法，没有消失使用常规匈牙利算法进行匹配  
![image](https://github.com/MakerCollider/20200623-haipeng_reid/blob/master/gif/2020.07.16_no_kalman200_00_30-00_00_39.gif)  
![image](https://github.com/MakerCollider/20200623-haipeng_reid/blob/master/gif/2020.07.16_no_kalman200_01_17-00_01_26.gif)  
### 2020/7/27
三人版本：2020-07-24_V5_Multi-REID_CNN.ipynb 
使用YOLOV5检测器代替之前的efficendet检测器   
使用交叉状态保持机制，避免因重叠导致的错误ID的情况，关闭kalman轨迹跟踪   
处理不同情况使用不同模式：1 None 2 Cross 3 Overlap 模式  
状态机制包含三个： 
inherit combine_split create  
继承 合并_分裂 构建  
![image](https://github.com/MakerCollider/20200623-haipeng_reid/blob/master/gif/2020.07.27_three_overlap_100_00_00-00_00_30.gif)  
### 2020/7/28
多人版本 2020-07-28_V5_Multi-REID_CNN.ipynb  
已测试四人  
![image](https://github.com/MakerCollider/20200623-haipeng_reid/blob/master/gif/demo_four.gif)  
### 2020/7/30   
New set CrossVideo 文件夹用于方便使用  
Reid=CNN_REID(  
                lib_dth=0.01,  
                passage_dth=0.02,  
                reid_dth=0.03,  
                forget=10,  
                kalman=False)  
    #arg1:库添加的阈值   
    #arg2:切换状态的阈值   
    #arg3:分辨不同人的阈值   
    #arg4:状态变为隐藏的帧数   
    #arg5:kalman开启的阀门  
 ### 2020/8/4  
 文件夹：MutilCrossVideo
 多视角~双视角的实现：  
 实现方法：  
 在原来的基础上：  
 改变ID管理：  
保存数据分为camera私有数据和共有数据  
包含的数据有：    '
camera':{},'lib':[],'frame':0,'passage':False,'hold':False  
包含的私有数据： 
'camera_id':{'bbox':[],'center':[],'kalman':[],'cross':False,'overlap':False,'overlap_id':[]}  
私有数据中center 使用的是同时出现相同ID的feature的中心构成  
![image](https://github.com/MakerCollider/20200623-haipeng_reid/blob/master/gif/2020.08.04_cross.gif)  
