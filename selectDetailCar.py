#coding:utf8#
#  -*-coding:utf8-*-#

__author__ = 'ASUS'

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

import os
import sys
import time
import numpy
import PIL
import cv2 as cv
import glob
import numpy as np
import cPickle
import string
import random

import trainDetailCNN_half_gray as trainCNN

# def load_data(dataset_path):
#
# 	read_file = open(dataset_path,'rb')
# 	datas,labels = cPickle.load(read_file)
# 	read_file.close()
#
# 	x, y = (datas,labels)
# 	return x, y

# def load_params(params_file):
#
# 	f=open(params_file,'rb')
# 	layer0_params_w,layer0_params_b=cPickle.load(f)
# 	layer1_params_w,layer1_params_b=cPickle.load(f)
# 	layer2_params_w,layer2_params_b=cPickle.load(f)
# 	layer3_params_w,layer3_params_b=cPickle.load(f)
# 	f.close()
#
# 	layer0_params = theano.shared(np.asarray(layer0_params_w,dtype=theano.config.floatX),borrow=True), \
# 					theano.shared(np.asarray(layer0_params_b,dtype=theano.config.floatX),borrow=True)
# 	layer1_params = theano.shared(np.asarray(layer1_params_w,dtype=theano.config.floatX),borrow=True), \
# 					theano.shared(np.asarray(layer1_params_b,dtype=theano.config.floatX),borrow=True)
# 	layer2_params = theano.shared(np.asarray(layer2_params_w,dtype=theano.config.floatX),borrow=True), \
# 					theano.shared(np.asarray(layer2_params_b,dtype=theano.config.floatX),borrow=True)
# 	layer3_params = theano.shared(np.asarray(layer3_params_w,dtype=theano.config.floatX),borrow=True), \
# 					theano.shared(np.asarray(layer3_params_b,dtype=theano.config.floatX),borrow=True)
#
# 	return layer0_params,layer1_params,layer2_params,layer3_params
#
# def load_params2(params_file):
#
# 	f=open(params_file,'rb')
# 	layer0_params=cPickle.load(f)
# 	layer1_params=cPickle.load(f)
# 	layer2_params=cPickle.load(f)
# 	layer3_params=cPickle.load(f)
# 	f.close()
#
# 	return layer0_params,layer1_params,layer2_params,layer3_params
#
# def shared_dataset(data_x, data_y, borrow=True):
# 	shared_x = theano.shared(np.asarray(data_x,
# 										   dtype=theano.config.floatX),
# 							 borrow=borrow)
# 	shared_y = theano.shared(np.asarray(data_y,
# 										   dtype=theano.config.floatX),
# 							 borrow=borrow)
# 	return shared_x, T.cast(shared_y, 'int32')
#

#
# def readAndSaveData(vehicleType,scale,addNegative = 0):
#
# 	#将所有种类的车辆样本集中到一个文件数据，并打乱其顺序
#
# 	dataDir = 'H:/veheicleSample/headOfCar/samplePKL/'
# 	# vehicleType = '3_benz'
# 	dataName = vehicleType+'01.pkl'
# 	# addNegative = 0
#
# 	# addNegative = 1
# 	# negaAddPath = dataDir +vehicleType+'_negative_add_half_gray.pkl'
#
#
# 	if ( not os.path.isfile(dataDir+dataName) or addNegative):
# 		# allVehicleType = ('0_vw','1_buick','2_audi','3_benz')
# 		# allOptType = ('train','test')
# 		# dataEmpty = True
# 		colorType = 'gray'
#
# 		# for vehicleType in allVehicleType:
# 		negativePath = dataDir + vehicleType +'_negative_half_gray.pkl'
# 		positivePath = dataDir + vehicleType +'_positive_half_gray.pkl'
#
# 		positive_x,positive_y =  load_data(positivePath)
# 		negative_x,negative_y =  load_data(negativePath)
#
# 		# for i in xrange (temp_test_x.shape[0]):
# 		# 	testImg = temp_test_x[i].reshape(34,34)
# 		# 	cv.imshow("testImg1",testImg)
# 		# 	cv.waitKey(0)
#
#
# 		# 集中数据
# 		# useNum = 2800
# 		# if dataEmpty:
# 		train_x =  np.row_stack((positive_x,negative_x))
# 		train_y =  np.append(positive_y,negative_y,axis = 0)
#
# 		# train_x =  np.row_stack((train_x,negatAdd_x))
# 		# train_y =  np.append(train_y,negatAdd_y,axis = 0)
#
# 		# 生成乱序1
# 		trainNum = train_x.shape[0]
# 		randomIndex = range(0,trainNum)
# 		random.shuffle(randomIndex)
#
# 		trainData_x = np.empty(train_x.shape,'uint8')
# 		trainData_y = np.empty(train_y.shape,'uint8')
#
# 		for i in range(trainNum):
# 			trainData_x[i] = train_x[randomIndex[i]]
# 			trainData_y[i] = train_y[randomIndex[i]]
#
# 		# scale = 34.0/49.0
# 		testData_x = trainData_x[trainData_x.shape[0] *scale:]
# 		testData_y = trainData_y[trainData_y.shape[0] *scale:]
# 		trainData_x = trainData_x[:trainData_x.shape[0] *scale]
# 		trainData_y = trainData_y[:trainData_y.shape[0] *scale]
#
# 		if 1 == addNegative:
# 			negaAddPath = dataDir +vehicleType+'_negative_add_half_gray.pkl'
# 			negatAdd_x,negatAdd_y =  load_data(negaAddPath)
# 			train_x =  np.row_stack((trainData_x,negatAdd_x))
# 			train_y =  np.append(trainData_y,negatAdd_y,axis = 0)
#
# 			# 生成乱序2
# 			trainNum = train_x.shape[0]
# 			randomIndex = range(0,trainNum)
# 			random.shuffle(randomIndex)
#
# 			trainData_x = np.empty(train_x.shape,'uint8')
# 			trainData_y = np.empty(train_y.shape,'uint8')
#
# 			for i in range(trainNum):
# 				trainData_x[i] = train_x[randomIndex[i]]
# 				trainData_y[i] = train_y[randomIndex[i]]
#
#
#
# 		allDatas = [(trainData_x, trainData_y),(testData_x, testData_y)]
# 		out_dir =  dataDir
# 		if not os.path.exists(out_dir):
# 			os.mkdir(out_dir)
#
# 		saveFile = open(out_dir + dataName,'wb')
# 		cPickle.dump(allDatas,saveFile)
# 		saveFile.close()
#
# 	else:
# 		in_dir =  dataDir
# 		readFile = open(in_dir + dataName,'rb')
# 		allDatas = cPickle.load(readFile)
# 		readFile.close()
#
# ############################################
# 	##################################
# 		 ######################
# 	trainData_x,trainData_y = allDatas[0]
# 	testData_x,testData_y = allDatas[1]
#
# 	trainData_x =  np.asarray(trainData_x,dtype='float32')/256.0
#
#    #对样本进行ZCA白化
#
# 	# V,S,mean_x = pca.pca(trainData_x)
# 	# epsilon = 0.1
# 	# U = V.T
# 	# ZCAWhite = U.dot( np.diag((1.0/np.sqrt(S+epsilon))).dot(U.T))  #U*(1/sqrt(s+epsilon))*U'
#
#
#
# 	# num_data,dim = trainData_x.shape
# 	# dim = 800
# 	# x = trainData_x[0:dim]
# 	# x = x.T
# 	# mean_x = x.mean(axis = 0)
# 	# x0 = x - mean_x  #tile 整块扩展矩阵
# 	# sigma = numpy.dot(x0,x0.T)/num_data
# 	# U,S,V = numpy.linalg.svd(sigma)
# 	# epsilon = 0.1
# 	# ZCAWhite = U.dot( np.diag((1.0/np.sqrt(S+epsilon))).dot(U.T))  #U*(1/sqrt(s+epsilon))*U'
#
# 	# 保存白化矩阵，在测试时需要使用相同的白化矩阵对测试图片进行白化
# 	# saveFile = open('ZCA.pkl','wb')
# 	# cPickle.dump(ZCAWhite,saveFile)
# 	# saveFile.close()
#
# 	# trainData_x = np.dot( (trainData_x - trainData_x.mean()),ZCAWhite)  #白化
#
# 	x_mean = trainData_x.mean(0)
# 	saveFile = open('x_mean.pkl','wb')
# 	cPickle.dump(x_mean,saveFile)
# 	saveFile.close()
#
# 	trainData_x = trainData_x - x_mean
# 	trainData_y =  trainData_y.flatten()
#
# 	testData_x = np.asarray(testData_x,dtype='float32')/256.0
# 	# testData_x = np.dot((testData_x - testData_x.mean()) ,ZCAWhite)
# 	testData_x = testData_x - x_mean
# 	testData_y =  testData_y.flatten()
#
# 	train_set_x, train_set_y = shared_dataset(trainData_x,trainData_y)
# 	test_set_x, test_set_y = shared_dataset(testData_x,testData_y)
#
# 	rval = [(train_set_x, train_set_y), (test_set_x, test_set_y),	(test_set_x, test_set_y)]
# 	return rval
#
#
# #分类器，即CNN最后一层，采用逻辑回归（softmax）
# class LogisticRegression(object):
# 	def __init__(self, input,params_W,params_b,usePreParams, n_in, n_out):
# 		if usePreParams:
# 			self.W = params_W
# 			self.b = params_b
#
# 		else:
# 			self.W = theano.shared(
# 				value=numpy.zeros(
# 					(n_in, n_out),
# 					dtype=theano.config.floatX
# 					),
# 				name='W',
# 				borrow=True
# 				)
# 			self.b = theano.shared(
# 					value=numpy.zeros(
# 					(n_out,),
# 					dtype=theano.config.floatX
# 					),
# 				name='b',
# 				borrow=True
# 		)
# 		# s = input.get_value(borrow=True).shape
# 		self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
# 		self.y_pred = T.argmax(self.p_y_given_x, axis=1)
# 		self.params = [self.W, self.b]
#
# 	def negative_log_likelihood(self, y):
# 		return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
#
# 	def errors(self, y):
# 		if y.ndim != self.y_pred.ndim:
# 			raise TypeError(
# 				'y should have the same shape as self.y_pred',
# 				('y', y.type, 'y_pred', self.y_pred.type)
# 			)
# 		if y.dtype.startswith('int'):
# 			# return (T.mean(T.neq(self.y_pred, y)),T.sum(T.neq(self.y_pred, y)),T.sum(T.eq(self.y_pred, y)))
# 			return T.mean(T.neq(self.y_pred, y))
# 		else:
# 			raise NotImplementedError()
#
#
# class HiddenLayer(object):
# 	def __init__(self, rng, input,params_W,params_b,usePreParams, n_in, n_out, W=None, b=None,
# 				 activation=T.tanh):
#
# 		self.input = input
# 		if usePreParams:
# 			self.W = params_W
# 			self.b = params_b
# 		else:
# 			if W is None:
# 				W_values = numpy.asarray(
# 					rng.uniform(
# 						low=-numpy.sqrt(6. / (n_in + n_out)),
# 						high=numpy.sqrt(6. / (n_in + n_out)),
# 						size=(n_in, n_out)
# 					),
# 					dtype=theano.config.floatX
# 				)
# 				if activation == theano.tensor.nnet.sigmoid:
# 					W_values *= 4
# 				W = theano.shared(value=W_values, name='W', borrow=True)
#
# 			if b is None:
# 				b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
# 				b = theano.shared(value=b_values, name='b', borrow=True)
#
# 			self.W = W
# 			self.b = b
#
# 		lin_output = T.dot(input, self.W) + self.b
# 		self.output = (
# 			lin_output if activation is None
# 			else activation(lin_output)
# 		)
# 		# parameters of the model
# 		self.params = [self.W, self.b]
#
#
#
# class LeNetConvPoolLayer(object):
#
# 	def __init__(self, rng, input ,params_W,params_b,usePreParams,filter_shape, image_shape, poolsize=(2, 2)):
#
# 		assert image_shape[1] == filter_shape[1]
# 		self.input = input
#
# 		fan_in = numpy.prod(filter_shape[1:])
# 		fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
# 				   numpy.prod(poolsize))
#
# 		if usePreParams:
# 			self.W = params_W
# 			self.b = params_b
#
# 		else:
# 			# initialize weights with random weights
# 			W_bound = numpy.sqrt(6. / (fan_in + fan_out))
# 			self.W = theano.shared(
# 				numpy.asarray(
# 					rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
# 					dtype=theano.config.floatX
# 				),
# 				borrow=True
# 			)
#
# 			# the bias is a 1D tensor -- one bias per output feature map
# 			b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
# 			self.b = theano.shared(value=b_values, borrow=True)
#
# 		# 卷积
# 		conv_out = conv.conv2d(
# 			input=input,
# 			filters=self.W,
# 			filter_shape=filter_shape,
# 			image_shape=image_shape
# 		)
#
# 		# 子采样
# 		pooled_out = downsample.max_pool_2d(
# 			input=conv_out,
# 			ds=poolsize,
# 			ignore_border=True
# 		)
#
# 		def ReLU(x):
# 			return theano.tensor.switch(x<0, 0, x)
# 		self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
# 		# self.output = T.nnet.sigmoid(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
# 		# self.output = ReLU(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
#
#
# 		# store parameters of this layer
# 		self.params = [self.W, self.b]
#
# 	def ReLU(x):
# 		return theano.tensor.switch(x<0, 0, x)
#
# def save_params(fileName,param1,param2,param3,param4):
# 		import cPickle
# 		write_file = open(fileName, 'wb')
# 		cPickle.dump(param1, write_file, -1)
# 		cPickle.dump(param2, write_file, -1)
# 		cPickle.dump(param3, write_file, -1)
# 		cPickle.dump(param4, write_file, -1)
# 		write_file.close()

###########################################
###############################
####################
############
######
def selectCar( threshold = 200):

	srcImgPath = 'H:\\veheicleSample\\headOfCar\\2-3.5\\audi\\12\\20150511_111529_0164_head.bmp'
	# dir_in = 'H:\\veheicleSample\\headOfCar\\2-3.5\\audi\\all\\'
	dir_in = 'H:\\veheicleSample\\headOfCar\\511\\'
	dir_out = 'H:\\veheicleSample\\headOfCar\\2-3.5\\audi\\selected'

	searchPath = dir_in

	ParamsPath = 'I:\\DL\\DL_py\\Car_project\\detailCarReco\\params\\nbparams\\'
	ParamsName = '90class_params5.pkl'

	mean_dir = 'H:\\veheicleSample\\headOfCar\\samplePKL\\detailClass'


	# Nclass = 90
	# 设置参数
	#输入图像尺寸
	resize_row = 98
	resize_col = 174


	imgNChannels = 1
	imgRows = 98
	imgCols = 98

	nkerns=[20,30]
	L0_imgRows = imgRows
	L0_imgCols = imgCols

	filterSize = [11,7]  # 卷积核宽度
	L0PoolSize = (4,4)
	L1PoolSize = (2,2)
	HL_nout = 1000  #隐层输入
	classNum = 90  #分类数量

	x = T.matrix('x')

	print '...building the model'

	layer0_params,layer1_params,layer2_params,layer3_params= \
		trainCNN.load_params(ParamsPath+ParamsName)

	rng = numpy.random.RandomState(23455)
	L0_imgShape = (1,imgNChannels,L0_imgRows,L0_imgCols)
	layer0_input = x.reshape(L0_imgShape)
	layer0 = trainCNN.LeNetConvPoolLayer(
		rng,
		input = layer0_input,
		params_W=layer0_params[0],
		params_b=layer0_params[1],
		usePreParams=True,
		image_shape = L0_imgShape,
		filter_shape = (nkerns[0],imgNChannels,filterSize[0],filterSize[0]),
		poolsize=L0PoolSize
	)


	# 第二个卷积+maxpool层,输入是上层的输出，即(batch_size, nkerns[0], 26, 21)
	L1_imgShapeRows = (L0_imgRows - filterSize[0]+1)/L0PoolSize[0]
	L1_imgShapeCols = (L0_imgCols - filterSize[0]+1)/L0PoolSize[0]
	layer1 = trainCNN.LeNetConvPoolLayer(
		rng,
		input=layer0.output,
		params_W=layer1_params[0],
		params_b=layer1_params[1],
		usePreParams=True,
		image_shape=(1, nkerns[0],L1_imgShapeRows,L1_imgShapeCols),
		filter_shape=(nkerns[1], nkerns[0], filterSize[1], filterSize[1]),
		poolsize=L1PoolSize
	)

	layer2_input = layer1.output.flatten(2)
	HL_nout = HL_nout
	L2_imgShapeRows = (L1_imgShapeRows - filterSize[1] + 1)/L1PoolSize[0]
	L2_imgShapeCols = (L1_imgShapeCols - filterSize[1] + 1)/L1PoolSize[0]
	layer2 = trainCNN.HiddenLayer(
		rng,
		input=layer2_input,
		params_W=layer2_params[0],
		params_b=layer2_params[1],
		usePreParams=True,
		n_in=nkerns[1] * L2_imgShapeRows * L2_imgShapeCols,
		n_out=HL_nout,      #全连接层输出神经元的个数，自己定义的，可以根据需要调节
		activation=T.tanh
	)

	layer3_input = layer2.output.flatten(2)   #(nkerns[2]),
	feature_vector = layer3_input

	layer3 = trainCNN.LogisticRegression(
		input=layer3_input,
		params_W=layer3_params[0],
		params_b=layer3_params[1],
		usePreParams=True,
		n_in=HL_nout, n_out=classNum)

	carRecog = theano.function(
		[x],
		layer3.y_pred,
	)


	carFeatureVect = theano.function(
		[x],
		feature_vector,
	)
	###############
	#测试
	###############
	read_file = open(mean_dir + '\\'+'x_mean90.pkl','rb')
	x_mean = cPickle.load(read_file)
	read_file.close()

	srcImg = cv.imread(srcImgPath,0)
	ResizeImg_temp = cv.resize(srcImg,(resize_col,resize_row))
	ResizeImg = ResizeImg_temp[: ,:resize_row]
	ResizeImg = cv.equalizeHist(ResizeImg)

	testData_x =  np.asarray(ResizeImg,dtype='float32').flatten(0)/256.0
	testData_x = (testData_x - x_mean).reshape((1,testData_x.shape[0]))

	srcVector = carFeatureVect(testData_x)


	for file in glob.glob(dir_in+'/*.bmp'):
		filePath,fileName = os.path.split(file)

		inputImg = cv.imread(file,0)
		ResizeImg_temp = cv.resize(inputImg,(resize_col,resize_row))
		ResizeImg = ResizeImg_temp[: ,:resize_row]
		ResizeImg = cv.equalizeHist(ResizeImg)

		testData_x =  np.asarray(ResizeImg,dtype='float32').flatten(0)/256.0
		testData_x = (testData_x - x_mean).reshape((1,testData_x.shape[0]))

		# destVector = carFeatureVect(testData_x)
		#
		# diff = 0
		# for k in xrange(destVector.shape[1]):
		# 	value = destVector[0,k] - srcVector[0,k]
		# 	if value*value > 0.001:
		# 		diff += value*value

		# if diff < threshold:  carRecog
		if 10 == carRecog(testData_x):
			deleteSrc = 1
			endP = string.rfind(fileName,'_')
			firstName = fileName[:endP]

			if (   deleteSrc	\
					and os.path.exists(dir_in + '/'+ firstName+'_result.jpg') \
					and os.path.exists(dir_in + '/'+firstName+'_src.jpg')\
					and os.path.exists(dir_in +'/'+ fileName)\
					and os.path.exists(dir_out)):

				imResult = cv.imread(dir_in +'/'+ firstName+'_result.jpg')
				imSrc = cv.imread(dir_in + '/'+firstName+'_src.jpg')
				imHead = cv.imread(dir_in + '/'+firstName+'_head.bmp')

				# 保存图片
				cv.imwrite(dir_out + '/'+ firstName+'_result.jpg',imResult)
				cv.imwrite(dir_out + '/'+firstName+'_src.jpg',imSrc)
				cv.imwrite(dir_out + '/'+firstName+'_head.bmp',imHead)

				# os.remove(dir_in + '/'+ firstName+'_result.jpg')
				# os.remove(dir_in + '/'+firstName+'_src.jpg')
				# os.remove(dir_in + '/'+firstName+'_head.bmp')

			# cv.imshow("head",testImg)
			# cv.imshow("result",imResult)
			# cv.imshow("src",imSrc)
			# cv.waitKey(0)
			# cv.imshow("result",testImg)










if __name__ == '__main__':

	selectCar()


