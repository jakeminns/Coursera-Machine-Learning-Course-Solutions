from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pylab
import scipy.io as sio

def linReg(theta,x):
	fin = np.empty([x.shape[0]])
	fin = theta*x
	for i in range(0,x.shape[0]):
		fin[i] = np.sum(x[i]*theta)
	return fin

def sigmoid(theta,x):
	fin = 1/(1+np.power(2.718281828459045,np.sum((-1*np.transpose(theta)*x),axis=1)))
	return fin
def sigmoidTest(theta,x):
	fin = 1/(1+np.power(2.718281828459045,np.sum((-1*np.transpose(theta)*x))))
	return fin
def costFuncLineaReg(m,features,y,theta):
	s = np.sum((linReg(theta,features)-y)**2)

	return (0.5*m)*s

def costFuncLinearRegDif(m,features,y,theta,xj):
	#Create empty array of size = xj (xj size = theta size) 
	fin = np.empty([xj.shape[0]])
	#Lopp through and calculate the partal derivitive for each theta d/d theta (sum (h_theta (xj) - y)*xj) xj = feature componant
	for i in range(0,xj.shape[0]):
		s = np.sum((linReg(theta,features)-y)*xj[i])
		fin[i] = s
	return fin

def costFuncLogisticReg(m,features,y,theta,lambda_1):
	sig = sigmoid(theta,features)
	logSig = np.log(sig)
	OneLogSig = np.log(1-sig)
	#log(1-x) tends to -inf at x = 1, python put this as -inf so filter to replace this with large number
	OneLogSig[OneLogSig==np.log(0)] = -10000000000000
	logSig[OneLogSig==np.log(0)] = -100000000000000
	#print("cost",logSig.shape,OneLogSig.shape)
	regularizationTerm = (lambda_1/(2*m))*np.sum(theta[1:])
	s = np.sum((-y*logSig)-(1-y)*OneLogSig)

	return (0.5*m)*s+regularizationTerm

def costFuncLogisticRegDif(m,features,y,theta,xj,lambda_1):
	fin = np.sum(np.transpose(xj)*np.transpose(sigmoid(theta,features)-np.transpose(y)),axis=0)
	regularizationTerm = (lambda_1/m)*np.sum(theta[1:])
	return fin+regularizationTerm

def classificaionContour(xmin,xmax,step,theta):
	x_arr = np.arange(xmin,xmax,step)
	features = np.insert(x_arr, 0, 1, axis=1)
	contour.append(sigmoidTest(theta,features))
	print("cont",contour)
	contour = np.transpose(np.reshape(contour,(20,20)))
	return contour

####Theta Construction
feat_num = 400
theta = np.ones((10,feat_num+1))
lambda_1 = 0

for i in range(0,10):
	mat_cont = sio.loadmat('ex3data1.mat')

	#plt.imshow(np.transpose(np.reshape(mat_cont['X'][3455],(20,20))), cmap='rainbow', interpolation='nearest')
	#plt.show()

	#Settings
	x = np.insert(mat_cont['X'], 0, 1, axis=1)

	#Starting Values
	theta_temp = np.ones(feat_num+1)

	featuresTrans = x
	features = np.transpose(featuresTrans)
	m = float(x.shape[0])

	alpha = 10

	y = mat_cont['y']

	print("Loop",i)
	## Generate features (x0,x1) array theta1*x_0+theta2*x_1

	#zero = 10 correction i n data matrix
	if i == 0:
		filter1 = 10
	else:
		filter1 = i

	y[y_master!=filter1] = 0
	y[y_master==filter1] = 1	

	#print(features)

	itera = []
	costList = []
	deviation = 100
	prevCost = costFuncLogisticReg(m,featuresTrans,y,theta[i],lambda_1)
	itter = 0
	cost = 10000000

	#while(abs(deviation)>0.0006 or cost <3000):
	for g in range(0,20):
		theta_temp = theta[i] - alpha*(1/m)*(costFuncLogisticRegDif(m,featuresTrans,y,theta[i],features,lambda_1))
		theta[i] = theta_temp
		itera.append(itter)
		itter+=1

		cost = costFuncLogisticReg(m,featuresTrans,y,theta[i],lambda_1)
		deviation = (((prevCost-cost)/prevCost))*100
		if(itter%2==0):
			print(itter,cost,deviation)

		prevCost = cost
		costList.append((cost))

	print("Number Training:",i)
	print("Test_one:",sigmoidTest(theta[i],featuresTrans[550]))
	print("Test_two:",sigmoidTest(theta[i],featuresTrans[1050]))
	print("Test_three:",sigmoidTest(theta[i],featuresTrans[1550]))
	print("Test_four:",sigmoidTest(theta[i],featuresTrans[2050]))
	print("Test_five:",sigmoidTest(theta[i],featuresTrans[2550]))
	print("Test_six:",sigmoidTest(theta[i],featuresTrans[3050]))
	print("Test_seven:",sigmoidTest(theta[i],featuresTrans[3550]))
	print("Test_eight:",sigmoidTest(theta[i],featuresTrans[4050]))
	print("Test_Nine",sigmoidTest(theta[i],featuresTrans[4550]))


