# encoding:utf-8

import tensorflow as tf
import numpy as np

import loadData
numDays = loadData.numDays

trainSamples = loadData._trainSamples
trainValues = loadData._trainValues

def getChunk(samples, values, chunkSize):
	stepStart = 0
	i = 0
	while stepStart < len(samples):
		stepEnd = stepStart + chunkSize
		if stepEnd < len(samples):
			yield i, samples[stepStart:stepEnd], values[stepStart:stepEnd]
			i+=1
		stepStart = stepEnd



class SimpleNetwork():
	def __init__(self, num_hidden, batch_size, learning_rate, learningLoops):
		"""
		初始化参数
		:param num_hidden:隐藏层神经元数量
		:param batch_size:一次训练个数
		"""
		self.batchSize = batch_size
		self.testBatchSize = 500
		# parameters
		self.numHidden = num_hidden
		self.learningRate = learning_rate
		self.learningLoops = learningLoops
		# graph
		self.graph = tf.Graph()
		self.tfTrainSamples = None      # 输入训练数据 参数（前n天价格）
		self.tfTrainValues = None       # 输入训练数据 值（收盘价）
		self.tfTrainPrediction = None   # 预测
		self.tfTestSamples = None       # 测试数据 参数（前n天价格）
		self.tfTestValues = None        # 测试数据 值（收盘价）
		self.tfTestPrediction = None    # 预测

		self.defineGraph()              # 初始化数据图
		self.session = tf.Session(graph=self.graph)
		self.writer = tf.train.SummaryWriter('./board', self.graph)

	def defineGraph(self):
		"""
		初始化数据图
		:return:
		"""
		with self.graph.as_default():
			with tf.name_scope('input'):
				# input
				self.tfTrainSamples = tf.placeholder(
					tf.float32, shape=(self.batchSize, numDays), name='trainSamples'
				)
				self.tfTrainValues = tf.placeholder(
					tf.float32, shape=(self.batchSize), name='trainValues'
				)
				# test
				self.tfTestValues = tf.placeholder(
					tf.float32, shape=(self.batchSize)
				)
				self.tfTestSamples = tf.placeholder(
					tf.float32, shape=(self.batchSize, numDays)
				)
			# 全链接层 full connected layers
			# 共2个隐藏层
			# variables
			with tf.name_scope('WeightsAndBiases'):
				fc1_weights = tf.Variable(
					tf.truncated_normal([numDays, self.numHidden], stddev=0.1), name='weight_fc1'
				)
				fc1_biases = tf.Variable(
					tf.constant(0.1, shape=[self.numHidden]), name='biases_fc1'
				)
				fc2_weights = tf.Variable(
					tf.truncated_normal([self.numHidden, self.numHidden], stddev=0.1), name='weight_fc2'
				)
				fc2_biases = tf.Variable(
					tf.constant(0.1, shape=[self.numHidden]), name='biases_fc2'
				)
				fc3_weights = tf.Variable(
					tf.truncated_normal([self.numHidden, 1], stddev=0.1), name='weight_fc3'
				)
				fc3_biases = tf.Variable(
					tf.constant(0.1, shape=[1]), name='biases_fc3'
				)
			# model computation
			def model(data):
				with tf.name_scope('hiddenlayer1'):
					hidden1 = tf.nn.sigmoid(tf.matmul(data, fc1_weights) + fc1_biases)
				with tf.name_scope('hiddenlayer2'):
					hidden2 = tf.nn.sigmoid(tf.matmul(hidden1, fc2_weights) + fc2_biases)
				with tf.name_scope('logits'):
					hidden3 = tf.matmul(hidden2, fc3_weights) + fc3_biases
				return hidden3

			logits = model(self.tfTrainSamples)
			self.loss = tf.reduce_mean(
				tf.pow((logits - self.tfTrainValues), 2)
			)
			self.optimizer = tf.train.GradientDescentOptimizer(self.learningRate).minimize(self.loss)
			self.trainPrediction = logits

	def run(self):
		"""
		运行时
		:return:
		"""
		with self.session as sess:
			tf.initialize_all_variables().run()     # 初始化参数
			for itrain in range(self.learningLoops):
				for i, samples, values in getChunk(trainSamples, trainValues, chunkSize = self.batchSize):
					_, l, predictions = sess.run(
						[self.optimizer, self.loss, self.trainPrediction],
						feed_dict={self.tfTrainSamples: samples, self.tfTrainValues: values}
					)
					accuracy = self.accuracy(predictions, values)
					if i % 10 == 0:
						print(str(i) + 'accuracy: ', accuracy)

	def accuracy(self, predictions, values):
		"""
		训练精度,平均误差
		:param predictions:
		:param values:
		:return:
		"""
		return 1.0 * np.sum(np.power((predictions-values), 2)) / predictions.shape[0]


if __name__ == '__main__':
	net = SimpleNetwork(num_hidden=15,batch_size=10,learning_rate=0.01,learningLoops=2)
	net.run()