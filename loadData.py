# encoding:utf-8
import numpy as np
numDays = 9     # 预测时间长度
dataPath = './data/data.txt'

datas = []

def loadData():
	"""
	读取数据
	:return:
	"""
	f = open(dataPath)
	_numSamples = 0
	for line in f:
		if line:
			data = float(line)
			if data > 0:
				_numSamples = _numSamples + 1
				datas.append(data)
	return _numSamples

def reformData():
	'''
	重构造数据
	:return:
	'''
	reform_samples = []
	reform_values = []
	for i in range(len(datas) - (numDays + 1)):
		temp_samples = []
		for j in range(numDays):
			temp_samples.append(datas[i + j])
		reform_samples.append(temp_samples)
		reform_values.append(datas[i + numDays])
	return np.array(reform_samples), np.array(reform_values)

loadData()
print ('样本数：'+str(len(datas)))

_trainSamples, _trainValues = reformData()
print ("训练样本X： ")
print (_trainSamples.shape)
print ("训练样本y： ")
print (_trainValues.shape)

