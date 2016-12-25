# -*- coding:utf-8 -*-

"""
Train the skip-gram for word vectors
"""


from __future__ import absolute_import

from __future__ import division

from __future__ import print_function



import os
import random
import tensorflow as tf
from collections import Counter,deque
import zipfile

import numpy as np



class DataAgent(object):
	def __init__(self,data_path,most_common_num):
		"""
		args:
			data_path, string: the absolute path of the training data, the corpus
			most_common_num, int: the number of most common words, other words will be reset to UNK, which means unknow
		"""
		self.data, self.count, self.word_id_dict, self.id_word_dict = self.extract_data(data_path,most_common_num)

		# data_index, indicate the word we are process
		self.data_index = 0

		#self.sampledTable = self.getSampleTable()

	def extract_data(self,data_path,most_common_num):
		with open(data_path,'r') as f:
			# get all words
			words = f.read().split(' ')

			# count the number of each word, but the "UNK" should be determined later
			words_counter = Counter(words)
			count  = [["UNK",-1]]
			count.extend(words_counter.most_common(most_common_num-1))


			# build the word-id dict
			word_id_dict = dict()
			for word,_ in count:
				word_id_dict[word] = len(word_id_dict)

			# build data, we use the id to represent the word, so that we can train the model
			data = []
			NumUNK = 0
			for word in words:
				if word in word_id_dict:
					data.append(word_id_dict[word])
				else:
					# for those word which are not most common, we use UNK to represent
					# the Id of "UNK" is 0
					NumUNK +=1
					data.append(0)
			# determine the number of "UNK"
			count[0][1] = NumUNK

			# build the id - word dict
			id_word_dict = {wordId:word for word,wordId in word_id_dict.items()}

		return data,count,word_id_dict,id_word_dict

	def gen_batch(self,batch_size,num_context_word_for_center_word,context_window):
		"""

		we generate a batch every time, for example, "The cat jumped on the puddle", when "jumped" is the center word, then "cat"
		is the label of "jumped", 
		args:
			batch_size, int: number of samples in one batch
			num_context_word_for_center_word, int: number of context word will be sampled for current center word
			context_window, int: the size of context window, we will consider the 2*context_window+1 words each time

		return:
			batch: the center word list
			labels: list of word in the context of center word
		"""
		# we will use a fix-length queue to store all words in a sliding window, the queue 
		#will push in a new word and pop out the oldest word when the window is sliding from left to right

		assert batch_size % num_context_word_for_center_word == 0


		batch = np.ndarray(shape=(batch_size),dtype = np.int32)
		labels = np.ndarray(shape  = (batch_size), dtype = np.int32)


		# the number of words will process in a context
		span = 2 * context_window + 1

		word_queue = deque(maxlen = span)

		# init the queue
		for _ in xrange(span):
			word_queue.append(self.data[self.data_index])
			self.data_index = (self.data_index + 1)%len(self.data)

		#print(word_queue)
		# now we can get the center word use word_queuep[context_window]

		for i in xrange(batch_size // num_context_word_for_center_word):
			target = context_window
			avoid_index = [target]

			# we get certain number words in the context for the center word
			for j in xrange(num_context_word_for_center_word):
				while  target in avoid_index:
					target = random.randint(0,span-1)

				avoid_index.append(target)

				batch[i * num_context_word_for_center_word + j] = word_queue[context_window]
				labels[i * num_context_word_for_center_word + j] = word_queue[target]

			word_queue.append(self.data[self.data_index])
			#print(self.data_index)
			#print(word_queue)
			self.data_index = (self.data_index + 1) % len(self.data)

		return batch,labels


	def getSampleTable(self):
		"""
		we build the table for negative sampling in this function. We use the unigram and raise to power of 3/4 mentioned in Google's
		Distributed representation of word vectors. 

		Algo Descrition:
			Suppose we have the data and frequency of each word, then we can say that a word will be sampled more likely if it has higher
			frequency.
		"""
		wordCount = np.array(item[1] for item in self.count)
		wordFrequency = wordCount/np.sum(wordCount)

		wordFrequencyCum = np.cumsum(wordFrequency) * len(self.data)

		sampledTable = [0]* len(self.data)

		j = 0
		for i in range(len(sampledTable)):
			while i > wordFrequencyCum[j]:
				j += 1

			sampledTable[i] = j

		return sampledTable

	def negativeSample(self,K,target):
		"""
		get K negative sample which should not be equal to target
		"""
		avoid = [target]
		for k in range(K):
			candidate = self.sampledTable[np.random.randint(0,len(self.sampledTable)-1)]
			while candidate in avoid:
				candidate = self.sampledTable[np.random.randint(0,len(self.sampledTable)-1)]

			avoid.append[candidate]
		return avoid[1:]











class  Word2Vec(object):
	"""docstring for  Word2Vec"""
	def __init__(self, data_agent):
		super( Word2Vec, self).__init__()

		self.data_agent = data_agent

		self.build_options()

		self.graph = self.build_graph()

	def build_options(self):
		self.vocabulary_size = 50000
		self.batch_size = 128
		self.embedding_size = 128
		self.context_window = 1
		self.num_context_word_for_center_word = 2
		self.num_negative_samples = 2

	def add_placeholder(self):
		self.batch = tf.placeholder(tf.int32,shape = [self.batch_size])

		# we compress the word in the context and negative words together
		self.labels = tf.placeholder(tf.int32,shape = [self.batch_size])

	def build_graph(self):
		graph = tf.Graph()
		with graph.as_default():
			self.add_placeholder()

			input_vectors = tf.Variable(tf.truncated_normal([self.vocabulary_size,self.embedding_size]))

			output_vectors = tf.Variable(tf.truncated_normal([self.vocabulary_size,self.embedding_size]))

			input_embed = tf.nn.embedding_lookup(input_vectors,self.batch)

			target_embed = tf.nn.embedding_lookup(output_vectors,self.labels)


			neg_sampled_ids,_,_ = tf.nn.fixed_unigram_candidate_sampler(
				true_classes = tf.reshape(tf.cast(self.labels,tf.int64),[self.batch_size,1]),
				num_true = 1,
				num_sampled = self.num_negative_samples,
				unique = True,
				range_max = self.vocabulary_size,
				distortion = 0.75,
				unigrams = [word_count[1] for word_count in self.data_agent.count])

			neg_sampled_embed = tf.nn.embedding_lookup(output_vectors,neg_sampled_ids)

			true_logits = tf.reduce_sum(tf.mul(input_embed,target_embed),axis =1)


			# we use the same negative samples for every sample in a batch
			neg_logits = tf.matmul(input_embed,neg_sampled_embed,transpose_b = True)


			true_ten = tf.nn.sigmoid_cross_entropy_with_logits(true_logits,tf.ones_like(true_logits))

			neg_ten = tf.nn.sigmoid_cross_entropy_with_logits(neg_logits,tf.zeros_like(neg_logits))
			
			#self.loss = tf.reduce_sum(true_ten)/self.batch_size

			self.loss = (tf.reduce_sum(true_ten)
			        + tf.reduce_sum(neg_ten))/self.batch_size

			optimizer = tf.train.GradientDescentOptimizer(0.01)

			self.train_op = optimizer.minimize(self.loss)

		return graph

	def train(self):
		with self.graph.as_default():
			init = tf.global_variables_initializer()

			with tf.Session() as sess:
				sess.run(init)

				for i in range(100001):
					batch,labels = self.data_agent.gen_batch(self.batch_size,self.num_context_word_for_center_word,self.context_window)
					loss,_= sess.run([self.loss,self.train_op],feed_dict = {self.batch:batch,self.labels:labels})
					if i % 10000 == 0:
						print("step:%s loss:%s"%(i,loss))



if __name__ == "__main__":
	da = DataAgent(r"/Users/hu/development/tensorflow/tensorflow_word_2_vec/text8",50000)
	#batch,label =  da.gen_batch(128,2,1)

	#print( max(da.data))
	#print(len(da.count))
	#print(batch)
	#print(label)
	#print(batch)
	#print(label)
	wordVec = Word2Vec(da)
	wordVec.train()













			



		

		





