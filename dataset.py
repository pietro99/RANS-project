import math
import sys
import numpy as np
import torch 
from torch.utils.data import Dataset
from tqdm import tqdm
from numpy import linalg as LA
import hashlib
import io
import os

v1 = 0.2
eps = 0.2
zeta = 3.0
def l1(ua):
    return abs((2 * v1 * math.log(eps)) / (math.sqrt(abs(ua)**2 + 4*v1*zeta) - abs(ua)))

def l2():
    return abs(math.sqrt(v1/zeta)*math.log(eps))

def l1Vector(ua_vect):
    small = sys.float_info.min
    ua_mean = LA.norm(ua_vect)+small
    unit_vect = ua_vect / ua_mean
    length = l1(ua_mean)
    return  np.array(unit_vect * length)
    
    
def l2Vector(ua_vect):
    small = sys.float_info.min
    ua_mean = LA.norm(ua_vect) + small
    unit_vect = ua_vect / ua_mean
    length = l2()
    vect = torch.tensor(unit_vect * length)
    return  rotate(-90,vect).numpy()

def centeredL1(l1Vect, coord):
    distx = abs(l1Vect[0] - coord[0])
    disty = abs(l1Vect[1] - coord[1])
    newVect = [l1Vect[0] + distx, l1Vect[1]+disty]
    shiftedVect = [coord, newVect]
    return newVect

def centeredL2(l1Vect, coord):
    distx = abs(l1Vect[0] - coord[0])
    disty = abs(l1Vect[1] - coord[1])
    newVect = [l1Vect[0] + distx, l1Vect[1]+disty]
    shiftedVect = [coord, np.rot90([newVect]).reshape(2,).tolist()]
    torchVector = torch.tensor(newVect)
    return rotate(-90,torchVector).tolist()


def rotate(deg, matrix):
    phi = torch.tensor(deg * math.pi / 180)
    s = torch.sin(phi)
    c = torch.cos(phi)
    rot = torch.stack([torch.stack([c, -s]),
                       torch.stack([s, c])])
    return  matrix.float() @ rot.t().float()

def angleDegrees(vector1, vector2):
    small = sys.float_info.min
    unit_vector_1 = vector1 / (np.linalg.norm(vector1)+small)
    unit_vector_2 = vector2 / (np.linalg.norm(vector2)+small)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    return (angle * (180/math.pi))

def angleGrad(vector1, vector2):

    small = sys.float_info.min
    unit_vector_1 = vector1 / (np.linalg.norm(vector1)+small)
    unit_vector_2 = vector2 / (np.linalg.norm(vector2)+small)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    return angle


def calculateAngle(vector):
    unit_vector_1 = vector[0] / np.linalg.norm(vector[0])
    unit_vector_2 = [1,0] / np.linalg.norm([1,0])
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    diameter1 = LA.norm(vector[0])/100
    diameter2 = LA.norm(vector[1])/100
    return (angle * (180/math.pi))#, diameter1, diameter2

def checkPoint(coordinate,center,l1l2):
    a = LA.norm(l1l2[0])
    b = LA.norm(l1l2[1])
    if a == 0 or b == 0: 
        print(f'a: {a} | b: {b}')
        print(center)
        print(l1l2)
    angle = angleGrad(l1l2[0], [1,0])
    x = coordinate[0]
    y = coordinate[1]
    h = center[0]
    k = center[1]
    return(((x-h)*math.cos(angle) + (y-k)*math.sin(angle))**2 / a**2) + (((x-h)*math.sin(angle) - (y-k)*math.cos(angle))**2 / b**2) <= 1


import hashlib
import io
import os
import sys
import pandas as pd
class SimulationData(Dataset):
    def __init__(self, directory="./data/data_file", fileName="aw=1.3_internal.csv", stencilNum=10, samplePerStencil=10, override=False):
        a = pd.read_csv(f'{directory}/{fileName}')
        filter_cols = ["Cx", "Cy", "Cz", "Ux", "Uy", "Uz", "V", "epsilon", "k", "mag(U)", "nut", "p", "yPlus", "Sfn", "Points:0", "Points:1", "Points:2"]
        self.xy = a[filter_cols].values
        self.initialCleanup()
        np.random.seed(1234)
        np.random.shuffle(self.xy)
        self.y = self.normalize(self.xy[:,[5,6,8]])
        self.x = self.xy[:,[0,1,2,3,4,7,9,10,11,12,13]]
        self.original_x = self.x
        self.original_y = self.y
        if stencilNum == -1:
            stencilNum = self.num_samples
        self.generateData(stencilNum,samplePerStencil, directory, fileName, override)
        
        
    def generateData(self, stencilNum, samplePerStencil, directory, fileName, override):
        h = self.generateHash(self.xy, stencilNum, samplePerStencil)
        path = self.checkIfHashExist(h, directory)
        if path and not override:
            tensors = torch.load(path)
            self.x = tensors["x"]
            self.y = tensors["y"]
            self.centers = tensors["centers"]
            return
        
        newX = []
        newY = []
        newCenter = []
        
        for i in tqdm(range(stencilNum)):
            cloudPoints, y, center_point = self.calculateDataForPointCenter(i,samplePerStencil)
            if not cloudPoints:
                i-=1
                continue
            newY.append(y)
            newX.append(cloudPoints)
            newCenter.append(center_point)
            
      
        self.x = torch.tensor(np.array(newX))
        self.y = torch.tensor(np.array(newY))
        self.centers = torch.tensor(np.array(newCenter))

        torch.save({"x":self.x, "y":self.y, "centers":self.centers}, f'{directory}/{fileName}_{stencilNum}_{samplePerStencil}.{h}.t')
        
    def calculateDataForPointCenter(self, index, samplePerStencil):
        center_point = self.getCoordinates(index)
        center_velocity = self.getVelocity(index)
        l1l2 = self.getL1andL2Vectors(index)
        if LA.norm(l1l2[0]) == 0: return False, False, False
        data = []
        y = self.y[index]
        sample_counter = 0
        for i in range(self.xy.shape[0]):
            point_coordinates = self.getCoordinates(i)
            is_in_cloud = checkPoint(point_coordinates,center_point,l1l2)
            if is_in_cloud:
                if(sample_counter >= samplePerStencil):
                    return data, y, center_point
                sample_counter += 1
                velocity = self.getVelocity(i)
                x_coord, y_coord = self.calculateRelativeCoordinates(center_point, point_coordinates)
                vx, vy = self.calculateRelativeSpeed(center_velocity, velocity)
                Cxyz = self.getC(i)
                C = self.getC(i)
                V = self.getV(i)
                P = self.getP(i)
                S = self.getS(i)
                magV = self.getMagV(i)
                data.append([ *Cxyz,vx, vy,V,P,magV,S, x_coord, y_coord])
        return False, False, False
    
    def calculateRelativeCoordinates(self, center, other):
        x_center = center[0].item()
        y_center = center[1].item()
        x = other[0].item()
        y = other[0].item()
        new_x = x-x_center / LA.norm(center)
        new_y = y-y_center / LA.norm(center)
        return new_x, new_y
    
    def calculateRelativeSpeed(self, center, other):
        x_center = center[0].item()
        y_center = center[1].item()
        x = other[0].item()
        y = other[0].item()
        new_x = x-x_center 
        new_y = y-y_center
        return new_x, new_y
 
    def generateHash(self, xy, param1, param2):
        h = hashlib.sha256(xy.tobytes())
        h.update(param1.to_bytes(5, 'big'))
        h.update(param2.to_bytes(5, 'big'))
        return h.hexdigest()
    
    def checkIfHashExist(self, h, directory):
        directory = os.fsencode(directory)
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if h in filename.split("."):
                path = os.path.join(directory.decode("utf-8")+"/", filename)
                return path
        return False

        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
        
    def __len__(self):
        return self.x.shape[0]
    
    def getP(self, index):
        return self.x[index][6]
    
    def getC(self, index):
        return self.x[index][0:2]
    
    def getV(self, index):
        return self.x[index][4]
    
    def getMagV(self, index):
        return self.x[index][5]
        
    def getCoordinates(self, index):
        return self.x[index][9:11]
    
    def getVelocity(self, index):
        return self.x[index][3:5]
    
    def getS(self, index):
        return self.x[index][8]
    
    def getVelocityVector(self):
        return self.x[:, [3,4]]
    
    def getCoordinateVector(self):
        return self.x[:,[9,10]]
    
    def getAllEllipsesAxes(self):
        all_l1l2 = []
        for x in tqdm(range(0,len(self.x))):
            a = self.getL1andL2Vectors(x).tolist()   
            all_l1l2.append(a)
        return np.array(all_l1l2)
    
    def getL1andL2Vectors(self, index):
        velocity = self.getVelocity(index).tolist()
        position = self.getCoordinates(index).tolist()
        l1Vect = l1Vector(velocity).tolist()
        l2Vect = l2Vector(velocity).tolist()
        #l1 = centeredL1(l1Vect, position)
        #l2 = centeredL2(l2Vect, position)
        return np.array([l1Vect, l2Vect])
    
    def initialCleanup(self):
        new_array = []
        for row in self.xy:
            if(row[-1] == 0):
                new_array.append(row.tolist())
        self.new_array = np.array(new_array)
        self.xy = np.delete(self.new_array,[2,5,16],1)
        
    def normalizeData(self):
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        
        self.transformers = []
        for i in range(self.x.shape[2]):
            transformer = StandardScaler()
            self.x[:,:,i] = torch.tensor(transformer.fit_transform(self.x[:,:,i]))
            self.transformers.append(transformer)
        self.label_transformer = StandardScaler()
        self.y = torch.tensor(self.label_transformer.fit_transform(self.y))

            
    def back(self):
        for i in  range(self.x.shape[2]):
            transformer = self.transformers[i]
            self.x[:,:,i] = torch.tensor(transformer.inverse_transform(self.x[:,:,i]))
        self.y = torch.tensor(self.label_transformer.inverse_transform(self.y))   

    def normalize(self, y):
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        a = scaler.fit_transform(y)
        return a    