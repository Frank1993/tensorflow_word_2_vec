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

from six.moves import xrange


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

	def gen_batch(self, batch_size,num_context_word_for_center_word,context_window):
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
		labels = np.ndarray(shape  = (batch_size,1), dtype = np.int32)


		# the number of words will process in a context
		span = 2 * context_window + 1

		word_queue = deque(maxLen = span)

		# init the queue
		for _ in xrange(span):
			word_queue.append(self.data[self.data_index])
			self.data_index = (self.data_index + 1)%len(self.data)

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

			word_queue.append(self.data_index)
			self.data_index = (self.data_index + 1) % len(self.data)

			return batch,labels







class  Word2Vec(object):
	"""docstring for  Word2Vec"""
	def __init__(self, data_agent):
		super( Word2Vec, self).__init__()

		self.data_agent = data_agent

	def build_options():
		self.batch_size = 128
		self.embedding_size = 128
		self.context_window = 1
		self.num_context_word_for_center_word = 2


	def build_graph():
		with tf.Graph():
			batch = tf.placeholder(tf.int32,shape = [self.batch_size])
			labels = tf.placeholder(tf.int32,shape = [self.batch_size,1])

			
		

		





