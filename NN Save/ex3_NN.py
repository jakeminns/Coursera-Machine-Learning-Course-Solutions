from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pylab
import scipy.io as sio

def sigmoid(theta,x):
	fin = 1/(1+np.power(2.718281828459045,np.sum((-1*theta*x),axis=1)))
	return fin

def costFuncLogisticReg(net_top,m,features,y,theta,lambda_1):
	a_lj = networkConstruct(net_top)
	s = []
	for img in range(0,len(features)-1):
		#Input
		x= (features[img])
		#x= x_data[img]
		
		#Forward Propergation
		a_lj = forwardProp(a_lj,theta,x)
		sig = a_lj[len(a_lj)-1]
		logSig = np.log(sig)
		OneLogSig = np.log(1-sig)
		#log(1-x) tends to -inf at x = 1, python put this as -inf so filter to replace this with large number
		OneLogSig[OneLogSig==np.log(0)] = -10000000000000
		logSig[OneLogSig==np.log(0)] = -100000000000000
		#print("cost",logSig.shape,OneLogSig.shape)
		regularizationTerm = (lambda_1/(2*m))*np.sum(theta[1:])
		s_i = np.sum((-y[img][1]*logSig)-(1-y[img][1])*OneLogSig)
		#s_i = np.sum((-y[img]*logSig)-(1-y[img])*OneLogSig)

		s.append(s_i)
	return (0.5*m)*np.sum(s)+regularizationTerm

def gradientChecking(features,theta,y,lambda_1,m,net_top):
	a_lj = networkConstruct(net_top)

	for img in range(0,5000):

		x = features[img]
		a_lj = forwardProp(a_lj,theta,x)
		
		for z in range(0,len(a_lj)):
			sig = a_lj[len(a_lj)-1]
			logSig = np.log(sig)
			OneLogSig = np.log(1-sig)
			#log(1-x) tends to -inf at x = 1, python put this as -inf so filter to replace this with large number
			OneLogSig[OneLogSig==np.log(0)] = -10000000000000
			logSig[OneLogSig==np.log(0)] = -100000000000000
			#print("cost",logSig.shape,OneLogSig.shape)
			regularizationTerm = (lambda_1/(2*m))*np.sum(theta[1:])
			s_i = np.sum((-y[img][1]*logSig)-(1-y[img][1])*OneLogSig)		



def classificaionContour(xmin,xmax,step,theta,net_top):
	contour = np.zeros(xmax*xmax)
	a_lj = networkConstruct(net_top)
	for x in range(xmin,xmax,1):
		for y in range(xmin,xmax,1):
			ar = np.array([1.0,x,y])
			contour[x] = forwardProp(a_lj,theta,ar)[len(net_top)-1]
	contour = np.transpose(np.reshape(contour,(100,100)))
	return contour

def buildTheta(net_top):
	#This is a list not numpy array may cause problems later.
	theta = []
	for i in range(1,len(net_top)):
		theta.append(np.random.rand(net_top[i],net_top[i-1]))
		theta[i-1] = np.insert(theta[i-1], 0, 1, axis=1)

		print(theta[i-1].shape)
	return theta

def numOutputPrediction(x):
	for i in range(0,len(x)):
		if(x[i]>0.7):
			return i+1
	return 'No Prediction Made'

def networkConstruct(net_top):
	a_lj = []
	#Input
	a_lj.append(np.ones(net_top[0]))
	#Hidden
	for i in range(1,len(net_top)-1):
		a_lj.append(np.ones(net_top[i]))
	#Output
	a_lj.append(np.ones(net_top[len(net_top)-1]))
	return a_lj
def outputTarget(y,net_top):
	out = np.zeros(net_top[len(net_top)-1])
	out[y-1]=1
	return out
def forwardProp(a_lj,theta,x):
	
	a_lj[0] = x 

	for i in range(1,len(theta)+1):
		#print("layer",i,theta[i-1].shape,x.shape)
		new_a_lj = sigmoid(theta[i-1],a_lj[i-1])
		#Add Bias to all except output
		if(i!=len(theta)):
			new_a_lj = np.insert(new_a_lj, 0, 1, axis=0)
		#Updte a_lj
		a_lj[i] = new_a_lj

	return a_lj	

def backProp(a_lj, y, theta,net_top,m,delta):

	#Delta

	#print("y",y)

	sig = []
	for i in range(1,len(net_top)):
		sig.append(np.zeros((net_top[i]-1)))

	dif = []
	for i in range(1,len(net_top)):
		dif.append(np.zeros((net_top[i],net_top[i-1],)))

	# -1 so using 0 works
	netLen = len(a_lj)-1 

	#Output layer
	sig[len(sig)-1] = (a_lj[netLen] - y)*np.multiply((a_lj[len(a_lj)-1]),(1-a_lj[len(a_lj)-1]))
	#print("SigIni",sig)
	#print("ThetIn",theta)
	#print("delta",delta)

	#Remove Bias layer in theta and a_lj (a_lj removal skips last entry as output has no bias)
	for i in range(len(theta)):
		theta[i] = np.delete(theta[i],0,axis=1)
	for i in range(0,len(a_lj)-1):
		a_lj[i] = np.delete(a_lj[i],0,axis=0)

	for l in range(len(sig)-1,0,-1):
		#((thetal^l)^T)*sig^(l+1)  here in the code l and l are the same because of the way theta is defined (there is no theta_0)
		a1 = (np.transpose(theta[l]).dot(sig[l]))
		#print("a1",(a1.shape),a_lj[l].shape,a1,a_lj[l])
		a2 =  np.multiply((a_lj[l]),(1-a_lj[l]))
		#print("a2",a2.shape,a2)

		a3 = np.multiply(a1 , (a2))
		#print("a3",a3.shape,a3)

		sig[l-1] = a3
	
	#print("deltaBB",delta)
	#print("SigBB",sig)
	#print("ABB",a_lj)


	for i in range(len(delta)-1,-1,-1):
		#print("DElta",i,len(a_lj),delta[i].shape,(sig[i]).shape ,(a_lj[i].shape))
		trans = np.transpose(a_lj[i])
		#print("MD0",trans,a_lj[i])

		#print("MD2:",np.outer(sig[i],trans))
		delta[i] = (delta[i] +np.outer(sig[i],trans))

	#print("deltaAA",delta)
	#print("thet",theta)




	#print("thetAA",theta)

	return theta,delta

###Input Data
mat_cont = sio.loadmat('ex3data1.mat')
mat_theta = sio.loadmat('ex3weights.mat')

#Insert adds 1 for x0 and theta0 componsnt (bias)
x_data = np.insert(mat_cont['X'], 0, 1, axis=1)
y_data = np.insert(mat_cont['y'], 0, 1, axis=1)

##Settings
feat_num = 400
m =feat_num
#Network topology [input+1 for bias,hidden layer+1 for bias,output]
net_top = [feat_num,25,10]

####Theta Construction
theta = buildTheta(net_top)
#theta[0] = mat_theta['Theta1']+0.1
#theta[1] = mat_theta['Theta2']+0.2
#Network Construction
a_lj = networkConstruct(net_top)
lambda_1 = 1

for i in range(0,100):
	count = 0

	delta = []
	for i in range(1,len(net_top)):
		delta.append(np.zeros((net_top[i],net_top[i-1],)))


	for img in range(0,5000):
		
		#Input
		
		x = x_data[img]
		#print("y",x)
		#Forward Propergation
		a_lj = forwardProp(a_lj,theta,x)

		prediction = numOutputPrediction(a_lj[len(a_lj)-1])
		#prediction = a_lj[len(a_lj)-1]
		label = y_data[img][1]
		theta,delta = backProp(a_lj, outputTarget(y_data[img][1],net_top), theta,net_top,m,delta)
		
		for i in range(0,len(theta),1):
			theta[i] = np.insert(theta[i], 0, 1, axis=1)
		#print(a_lj[len(a_lj)-1],label)
		if prediction==label:
			count+=1

		#print("Output",img,a_lj[len(a_lj)-1])

	alpha = 0.01

	#Update Theta 

	for i in range(0,len(theta),1):
			#for i in range(len(theta)):
		#print(theta[i].shape,delta[i].shape)
		theta[i] = np.delete(theta[i],0,axis=1)

		#print("die",theta[i].shape,delta[i].shape)
		
		regSum = lambda_1*np.sum(theta[i][1:len(theta[i])-1])
		#print(m,np.divide(1.0,m))
		#print(alpha*(1.0/m)*(delta[i]))
		#print("reg:",regSum)
		#Reg term shgould acount for j=0 and j!=0
		theta_temp = theta[i] - alpha*(1.0/m)*(delta[i]+regSum)

		theta[i] = theta_temp
	
	for i in range(0,len(theta),1):
		theta[i] = np.insert(theta[i], 0, 1, axis=1)
	
	#Cost Function
	
	
	print("Cost:",costFuncLogisticReg(net_top,m,x_data,y_data,theta,lambda_1),np.sum(mat_theta['Theta1']-theta[0]),np.sum(mat_theta['Theta2']-theta[1]),count)
