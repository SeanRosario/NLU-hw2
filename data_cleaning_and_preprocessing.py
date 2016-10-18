import numpy as np
import os
import re
import matplotlib.pyplot as plt
import ast
import pickle

vocab=[]

implement_bigrams = False

def clean_str(string):
	"""
	Tokenization/string cleaning for all datasets except for SST.
	Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
	"""
	string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
	string = re.sub(r"\'s", " \'s", string)
	string = re.sub(r"\'ve", " \'ve", string)
	string = re.sub(r"n\'t", " not", string)
	string = re.sub(r"\'re", " \'re", string)
	string = re.sub(r"\'d", " \'d", string)
	string = re.sub(r"\'ll", " \'ll", string)
	string = re.sub(r",", " , ", string)
	string = re.sub(r"!", " ! ", string)
	string = re.sub(r"\(", " \( ", string)
	string = re.sub(r"\)", " \) ", string)
	string = re.sub(r"\?", " \? ", string)
	string = re.sub(r"\s{2,}", " ", string)
	string = re.sub(r"<br />", " ", string) #Replace HTML break with white space
	string = re.sub(r"br", " ", string)
	string = re.sub(r"\\", " ", string)
	return string.strip().lower()


def replace_with_oov(input_str,vocab):
    result=''
    for word in input_str.split():
        if (word in vocab):
            result= result + word + ' '
        else:
            result= result + '<oov> '
    return result

def cut_it_at(input_str,cap=300):
    result=''
    count=1
    for word in input_str.split():
        if (count<=cap):
            result= result + word + ' '
        else:
            break
        count=count+1
    return result

def add_bi_grams(input_str):
    result=' '
    words = input_str.split()
    for i in range(len(words)-1):
        bi_word = words[i] + '-' + words[i+1]
        result= result + bi_word + ' '
    return input_str + result


def prepare_data_and_labels():
	"""
	Loads IMDB polarity data from files, splits the data into words and generates labels.
	Saves split sentences and labels using pickle.
	"""

	pos_dir = './aclImdb/train/pos/'
	neg_dir = './aclImdb/train/neg/'

	positive_examples = []
	negative_examples = []

	for _file in os.listdir(pos_dir):
		text_file = open('./aclImdb/train/pos/'+_file, "r")
		lines = text_file.read()
		positive_examples.append(lines)
		
	for _file in os.listdir(neg_dir):
		text_file = open('./aclImdb/train/neg/'+_file, "r")
		lines = text_file.read()
		negative_examples.append(lines)

	x_text = positive_examples + negative_examples
	x_text = [clean_str(sent) for sent in x_text]



	word_count = {} # Keys are words, Values are frequency

	for review in x_text:
		
		words = review.split()
		
		for word in words:
			try:
				word_count[word]+=1
			except:
				word_count[word]=0
		
			
	res = list(sorted(word_count, key=word_count.__getitem__, reverse=True))

	global vocab
	vocab = res[:10000]

	# Replacing words that are not in the vocab with '<oov>'
	cleaned_x_text = [replace_with_oov(item,vocab) for item in x_text]

	bigram_dict = {}
	for review in cleaned_x_text:
	    words = review.split()
	    for i in range(len(words)-1):
	        bi_word = words[i] + '-' + words[i+1]
	        try:
	            bigram_dict[bi_word]+=1
	        except:
	            bigram_dict[bi_word]=0


	bigram_res = list(sorted(bigram_dict, key=bigram_dict.__getitem__, reverse=True))
	bigram_vocab = bigram_res[:10000]

	bigrammed = []
	for review in cleaned_x_text:
	    words = review.split()
	    s=''
	    for i in range(len(words)-1):
	        bi_word = words[i] + '-' + words[i+1]
	        s=s+bi_word+' '
	    bigrammed.append(s)

	bigram_cleaned_x_text = [replace_with_oov(item,bigram_vocab) for item in bigrammed]

	new_thang = []

	for a,b in zip(cleaned_x_text,bigram_cleaned_x_text):
	    new_thang.append(a+b)



	positive_labels = [[0, 1] for _ in positive_examples]
	negative_labels = [[1, 0] for _ in negative_examples]
	y = np.concatenate([positive_labels, negative_labels])



	with open('x_text_file.txt', 'wb') as f1:
		if (implement_bigrams==True):
			pickle.dump(new_thang, f1)
		else:
			pickle.dump(cleaned_x_text, f1)

	f1.close()

	with open('y_file.txt', 'wb') as f2:
		pickle.dump(y, f2)

	f2.close()







