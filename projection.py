
# To get the 3D position(X,Y,Z) from stereo images 
# Project 3D position(X,Y,Z) to the third oblique image for cross validation
import os
import cv2
import xml.dom.minidom
from numpy import *
import numpy as np
import sift_match
import os,sys
import numpy as np
import xml.dom.minidom as DM


def getInt_para(xmlfile): 
	dom = xml.dom.minidom.parse(xmlfile)
	#得到文档元素对象
	root = dom.documentElement
	# ImageWidth = float(root.getElementsByTagName('Width')[0].childNodes[0].data)
	FocalLengthPixels = float(root.getElementsByTagName('FocalLengthPixels')[0].childNodes[0].data)
	# SensorSize = float(root.getElementsByTagName('SensorSize')[0].childNodes[0].data)
	# f = ImageWidth*FocalLength/SensorSize
	f = FocalLengthPixels
	PrincipalPoint = root.getElementsByTagName('PrincipalPoint')[0]
	x0 = float(PrincipalPoint.getElementsByTagName('x')[0].childNodes[0].data)
	y0 = float(PrincipalPoint.getElementsByTagName('y')[0].childNodes[0].data)
	# 相机内参
	K = mat([[f,0,x0],[0,f,y0],[0,0,1]]).reshape(3,3)
	Distortion = root.getElementsByTagName('Distortion')

	if Distortion == []:
		distortion = np.array([0,0,0,0,0])


	# print(root.getElementsByTagName('Distortion'))
	# exit(0)
	else:
	
		K1 = float(Distortion.getElementsByTagName('K1')[0].childNodes[0].nodeValue)
		K2 = float(Distortion.getElementsByTagName('K2')[0].childNodes[0].nodeValue)
		K3 = float(Distortion.getElementsByTagName('K3')[0].childNodes[0].nodeValue)
		P1 = float(Distortion.getElementsByTagName('P1')[0].childNodes[0].nodeValue)
		P2 = float(Distortion.getElementsByTagName('P2')[0].childNodes[0].nodeValue)
		distortion = np.array([K1,K2,K3,P1,P2])

	return K, x0, y0, distortion

def getExt_para(xmlfile, imgname):
	dom = xml.dom.minidom.parse(xmlfile)
	#得到文档元素对象
	root = dom.documentElement
	ImagePathlist = root.getElementsByTagName('ImagePath')

	for i, Path in enumerate(ImagePathlist):
		ImagePath = Path.childNodes[0].data
		ImageName = ImagePath.split('\\')[-1]

		if ImageName == imgname:
			M_00 = float(root.getElementsByTagName('M_00')[i].childNodes[0].data)
			M_01 = float(root.getElementsByTagName('M_01')[i].childNodes[0].data)
			M_02 = float(root.getElementsByTagName('M_02')[i].childNodes[0].data)
			M_10 = float(root.getElementsByTagName('M_10')[i].childNodes[0].data)
			M_11 = float(root.getElementsByTagName('M_11')[i].childNodes[0].data)
			M_12 = float(root.getElementsByTagName('M_12')[i].childNodes[0].data)
			M_20 = float(root.getElementsByTagName('M_20')[i].childNodes[0].data)
			M_21 = float(root.getElementsByTagName('M_21')[i].childNodes[0].data)
			M_22 = float(root.getElementsByTagName('M_22')[i].childNodes[0].data)
			# 旋转矩阵
			R = mat([[M_00,M_01,M_02],[M_10,M_11,M_12],[M_20,M_21,M_22]]).reshape(3,3)

			x = float(root.getElementsByTagName('x')[i+1].childNodes[0].data)
			y = float(root.getElementsByTagName('y')[i+1].childNodes[0].data)
			z = float(root.getElementsByTagName('z')[i].childNodes[0].data)
			# 相机中心		
			C_center=mat([x,y,z]).reshape(3,1)

	return R, C_center


def get_pixelPoint(vertice, R, center, K, distortion): #将三维模型中的空间点投影到影像中

    [k1, k2, k3, p1, p2] = distortion  #无畸变
    vertice = vertice[1:] if len(vertice)==4 else vertice
    vertice = np.array(vertice).reshape((3,1))
    center = center.reshape((len(center),1))

    T = -np.dot(R, center)
    #camera coordination
    XYZ = np.dot(R, vertice) + T  
    u_ = XYZ[0,0]/XYZ[2,0]
    v_ = XYZ[1,0]/XYZ[2,0]

    r_2 = u_* u_ + v_ * v_
    u = u_*(1+k1*r_2+k2*r_2**2+k3*r_2**3)+2*p1*u_*v_+p2*(r_2+2*u_*u_) #矫正后的
    v = v_*(1+k1*r_2+k2*r_2**2+k3*r_2**3)+2*p2*u_*v_+p1*(r_2+2*v_*v_)

    px = K[0,0]*u + K[0,2]
    py = K[1,1]*v + K[1,2]

    # print('****************************')
    print('px:', px)
    print('py:', py)

    return px, py

def calculate_3DX(kp1, kp2, Proj1, Proj2):
	A0 = mat(kp1[0] * Proj1[2,:] - Proj1[0,:])
	A1 = mat(kp1[1] * Proj1[2,:] - Proj1[1,:])
	A2 = mat(kp2[0] * Proj2[2,:] - Proj2[0,:])
	A3 = mat(kp2[1] * Proj2[2,:] - Proj2[1,:])

	train_data = mat(vstack((A0,A1,A2,A3)))
	U,sigma,VT = np.linalg.svd(train_data)
	posx = VT[3,:].T
	posx_ = posx / posx[3][0]
	position = posx_[0:3]

	return position


if __name__ == '__main__':
	# get camera intrinsic parameters K, C
	xmlfile = './Block_1 - AT -export.xml'
	
	
	#cross validation
	img1 = 'DSC00041.jpg'
	img2 = 'DSC00044.jpg'

	img3 = 'DSC00047.jpg'

	# img_patch1 = './testimg/TS_l1.png'  #queryImage
	# img_patch2 = './testimg/TS_l2.png' #trainingImage

	# queryImage = cv2.imread(img_patch1)
	# trainingImage = cv2.imread(img_patch2)

	K, x0, y0, distortion = getInt_para(xmlfile)

	# get camera external parameters
	R1, C_center1 = getExt_para(xmlfile, img1)
	R2, C_center2 = getExt_para(xmlfile, img2)
	R3, C_center3 = getExt_para(xmlfile, img3)

	t1 = -R1 * C_center1
	t2 = -R2 * C_center2

	Proj1 = mat(K*hstack((R1,t1)))
	Proj2 = mat(K*hstack((R2,t2)))

	img_kp1 =(7244, 2327) # Points in Oblique image
	img_kp2 =(7208, 2458) # Points in Oblique image
	position_3D = calculate_3DX(img_kp1, img_kp2, Proj1, Proj2) # Triangulation
	print('position:', float(position_3D[0][0]))

	
	vertice = np.array([float(position_3D[0][0]), float(position_3D[1][0]), float(position_3D[2][0])])
	print(vertice)

	# Projection from 3D to 2D 
	position_2D = get_pixelPoint(vertice, R3, C_center3, K, distortion)