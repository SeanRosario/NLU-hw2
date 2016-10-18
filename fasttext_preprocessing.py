import numpy as np
import os
import re
import matplotlib.pyplot as plt
import ast
import pickle
from random import shuffle

def correct_format(input_str):
    formatted_str = re.sub(r"\n", " ", input_str)
    formatted_str = re.sub(r"<br />", " ", input_str) #Replace HTML break with white space
    #formatted_str = re.sub(r"br", " ", input_str)
    return formatted_str


def prepare_data_and_labels():

    """
    Loads IMDB polarity data from files, splits the data into words and generates labels.
    Saves train and test into two separate txt files
    """

    pos_dir = './aclImdb/train/pos/'
    neg_dir = './aclImdb/train/neg/'

    positive_examples = []
    negative_examples = []



    for _file in os.listdir(pos_dir):
        text_file = open('./aclImdb/train/pos/'+_file, "r")
        lines = text_file.read()
        lines = correct_format(lines)
        positive_examples.append('__label__POS  '+lines +'\n')

    for _file in os.listdir(neg_dir):
        text_file = open('./aclImdb/train/neg/'+_file, "r")
        lines = text_file.read()
        lines = correct_format(lines)
        negative_examples.append('__label__NEG  '+lines +'\n')

    x_text = positive_examples + negative_examples

    shuffle(x_text)


    # Split train/test set

    x_train = ''
    for item in x_text[5000:]:
        x_train+=item

    x_dev = ''
    for item in x_text[:5000]:
        x_dev+=item

    with open('fastText/x_dev.txt', 'wb') as f1:
        pickle.dump(x_text[:5000], f1)

    text_file = open("fastText/train.txt", "w")
    text_file.write(x_train)
    text_file.close()

    text_file = open("fastText/test.txt", "w")
    text_file.write(x_dev)
    text_file.close()





if __name__ == "__main__":
    print("Preparing data and labels...")
    prepare_data_and_labels()
    print("Done with preparing data and labels!")
