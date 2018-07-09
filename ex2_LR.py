from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pylab

def linReg(theta,x):
	fin = np.empty([x.shape[0]])
	for i in range(0,x.shape[0]):
		fin[i] = np.sum(x[i]*theta)
	return fin

def sigmoid(theta,x):
	fin = np.empty([x.shape[0]])
	for i in range(0,x.shape[0]):
		fin[i] = 1/(1+np.power(2.718281828459045,(-1*np.sum(x[i]*theta))))
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
	regularizationTerm = (lambda_1/(2*m))*np.sum(theta[1:])
	s = np.sum((-y*logSig)-(1-y)*OneLogSig)

	return (0.5*m)*s+regularizationTerm

def costFuncLogisticRegDif(m,features,y,theta,xj,lambda_1):
	#Create empty array of size = xj (xj size = theta size) 
	fin = np.empty([xj.shape[0]])
	#Lopp through and calculate the partal derivitive for each theta d/d theta (sum (h_theta (xj) - y)*xj) xj = feature componant
	for i in range(0,xj.shape[0]):
		#print("HERE",sigmoid(theta,features),"yyyy",y,"xjjjjj",xj[i])
		s = np.sum((sigmoid(theta,features)-y)*xj[i])
		fin[i] = s
	regularizationTerm = (lambda_1/m)*np.sum(theta[1:])
	return fin+regularizationTerm

def classificaionContour(xmin,ymin,xmax,ymax,step,theta):
	x_arr = np.arange(xmin,xmax,step)
	y_arr = np.arange(ymin,ymax,step)
	contour = []

	for y in y_arr:
		item = []
		features = np.array([np.ones(len(x_arr)),x_arr,np.full(x_arr.shape,y),x_arr*x_arr,np.full(x_arr.shape,y)*np.full(x_arr.shape,y)])
		featuresTrans = np.transpose(features)
		contour.append(sigmoid(theta,featuresTrans))

	return contour


x,n,y = np.genfromtxt('ex2data1.txt',delimiter=',',unpack=True)
#fig, ax = plt.subplots()
plt.scatter(x,n,marker='o',linestyle='None',c=y)
plt.show()

#Settings
feat_num = 2
x = x
n = n
lambda_1 = 1

#Starting Values
theta = np.ones(feat_num+1)
theta_temp = np.ones(feat_num+1)
m = float(x.shape[0])
alpha = 0.0001
theta[0] = 0
theta[1] = 1
## Generate features (x0,x1) array theta1*x_0+theta2*x_1
features = np.array([np.ones(len(x)),x,n])
featuresTrans = np.transpose(features)

#print(features)

itera = []
costList = []
deviation = 100
prevCost = costFuncLogisticReg(m,featuresTrans,y,theta,lambda_1)
itter = 0
cost = 10000000

#while(abs(deviation)>0.0006 or cost <3000):
for g in range(0,1500):
	theta_temp = theta - alpha*(1/m)*(costFuncLogisticRegDif(m,featuresTrans,y,theta,features,lambda_1))
	theta = theta_temp

	itera.append(itter)
	itter+=1

	cost = costFuncLogisticReg(m,featuresTrans,y,theta,lambda_1)
	deviation = (((prevCost-cost)/prevCost))*100
	if(itter%100==0):
		print(itter,cost,deviation,theta)
	prevCost = cost
	costList.append(np.log(cost))


plt.plot(itera,costList,marker='o',linestyle='None')
plt.show()

contour = classificaionContour(0,0,1.0,1.0,0.01,theta)
plt.imshow(contour, cmap='rainbow', interpolation='nearest')
#line = -(theta[0]/theta[2]*100)-(theta[1]/theta[2])*x*100
print(len(x),len(n))
#plt.scatter(150+(x*100),150+(n*100),marker='o',linestyle='None',c=y)
#plt.plot(x*100,line,linestyle='-')

plt.show()
