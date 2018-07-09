from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def linReg(theta,x):
	fin = np.empty([x.shape[0]])
	for i in range(0,x.shape[0]):
		fin[i] = np.sum(x[i]*theta)
	return fin

def costFunc(m,features,y,theta):
	s = np.sum((linReg(theta,features)-y)**2)

	return (0.5*m)*s

def costFuncDif(m,features,y,theta,xj):
	#Create empty array of size = xj (xj size = theta size) 
	fin = np.empty([xj.shape[0]])
	#Lopp through and calculate the partal derivitive for each theta d/d theta (sum (h_theta (xj) - y)*xj) xj = feature componant
	for i in range(0,xj.shape[0]):
		s = np.sum((linReg(theta,features)-y)*xj[i])
		fin[i] = s
	return fin

x0,x1,y = np.genfromtxt('ex1data2.txt',delimiter=',',unpack=True)

#basic feature scaling
x1=x1*1000

#Polynomial Testing
x = np.arange(0,1.3,0.001)
xSq = x*x
xCub = x*x*x
y =  0.132 + (0.1664*x)-(0.812*xSq)+0.9*xCub
noise = np.random.normal(0,0.05,len(y))
y = y+noise
plt.plot(x,y,marker='o',linestyle='None',color='red')
plt.show()

#Settings
feat_num = 3

#Starting Values
theta = np.ones(feat_num+1)
theta_temp = np.ones(feat_num+1)
m = float(x0.shape[0])
alpha = 0.01

#Feature Scaling
x_min = np.amin(x)
x_max = np.amax(x)
x_av = np.mean(x)

x_sc = (x-x_min)/(x_max-x_min)

xSq_min = np.amin(xSq)
xSq_max = np.amax(xSq)
xSq_av = np.mean(xSq)

xSq_sc = (xSq-xSq_min)/(xSq_max-xSq_min)

## Generate features (x0,x1) array theta1*x_0+theta2*x_1
features = np.array([np.ones(len(x)),x,xSq,xCub])
featuresTrans = np.transpose(features)


itera = []
costList = []
deviation = 100
prevCost = costFunc(m,featuresTrans,y,theta)
itter = 0
cost = 1000

while(abs(deviation)>0.001 or cost <1):

	theta_temp = theta - alpha*(1/m)*(costFuncDif(m,featuresTrans,y,theta,features))
	theta = theta_temp

	itera.append(itter)
	itter+=1

	cost = costFunc(m,featuresTrans,y,theta)
	deviation = (((prevCost-cost)/prevCost))*100
	print(cost,deviation,theta)
	prevCost = cost
	costList.append(np.log(cost))

print(theta)

plt.plot(itera,costList,marker='o',linestyle='None')
plt.show()

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(x0,x1,y)

line = theta[0]+theta[1]*x+theta[2]*xSq+theta[3]*xCub
#ax.plot(line)

plt.show()

plt.plot(x,y,marker='o',linestyle='None',color='red')

plt.plot(x,line,marker='*',linestyle='None')

plt.show()