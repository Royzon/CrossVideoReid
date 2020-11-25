import cv2
import numpy as np
class KalmanFilter(object):
    def __init__(self,bbox):
        (x,y,w,h)=bbox
        self.x,self.y=0,0
        self.last_measurement = self.current_measurement = np.array((2,1),np.float32)
        np_bbox=np.asarray([x,y]).astype(np.float32)
        self.last_predicition = self.current_prediction = np_bbox.resize((2,1))#np.zeros((2,1),np.float32)
        self.kalman = cv2.KalmanFilter(4, 2)
        #设置测量矩阵
        self.kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
        #设置转移矩阵
        self.kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
        #设置过程噪声协方差矩阵
        self.kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32)*0.003
    def move(self,x,y):
        #初始化
        self.last_measurement = self.current_measurement
        self.last_prediction = self.current_prediction
        #传递当前测量坐标值
        self.current_measurement = np.array([[np.float32(x)],[np.float32(y)]])
        #用来修正卡尔曼滤波的预测结果
        self.kalman.correct(self.current_measurement)
        # 调用kalman这个类的predict方法得到状态的预测值矩阵，用来估算目标位置
        current_prediction = self.kalman.predict()
        #上一次测量值
        lmx,lmy = self.last_measurement[0],self.last_measurement[1]
        #当前测量值
        cmx,cmy = self.current_measurement[0],self.current_measurement[1]
        #上一次预测值
        #lpx,lpy = last_prediction[0],last_prediction[1]
        #当前预测值
        cpx,cpy = current_prediction[0],current_prediction[1]
        return cpx,cpy
    def update(self,bbox):
        (x,y,w,h)=bbox
        self.x,self.y = self.move(x,y)
    def get(self):
        return self.x,self.y
if __name__ == '__main__':
	kalman=KalmanFilter([0,0,0,0])