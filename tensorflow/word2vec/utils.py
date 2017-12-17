from __future__ import print_function
import os
import time
import random
import zipfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from multiprocessing import Process

from scipy.spatial.distance import cosine
from six.moves import urllib
from collections import Counter
from nltk.corpus import stopwords
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


DownloadUrl = 'http://mattmahoney.net/dc/'
FileName = 'text8.zip'
DataFolder = 'data/'
VisualizeWordSize = 2000
VisualizeWordFile = 'data/Visualwords.txt'
EmbedVectorFile = 'data/embed_vectors.txt'


def download(file_name):
    file_path = os.path.join(DataFolder, FileName)
    if os.path.exists(file_path):
        print("{} already present".format(file_name))
        return file_name
    file_name, _ = urllib.request.urlretrieve(DownloadUrl+file_name, file_path)
    file_stat = os.stat(file_path)
    print("{0} of size {1} downloaded".format(file_name, file_stat.st_size))
    return file_name


def read_data(file_path):
    with zipfile.ZipFile(file_path) as fr:
        words = tf.compat.as_str(fr.read(fr.namelist()[0])).split(' ')
    return words


def preprocess(data_words):
    stop = stopwords.words('english')
    stop.extend(list('abcdefghijklmnopqrstuvwxyz'))
    stop.append('')
    stop = set(stop)
    return [word for word in data_words if word not in stop]


def build_vocab(words, vocab_size):
    counts = [('lcw', -1)]
    counts.extend(Counter(words).most_common(vocab_size - 1))
    index = 0
    word_dict = {}
    with open(VisualizeWordFile, 'w') as fr:
        for word, count in counts:
            if word is None:
                continue # to counter none words
            word_dict[word] = index
            if True:
                fr.write(word)
                fr.write("\n")
            index += 1
    return word_dict


def assign_index_to_words(words, words_dict):
    return [words_dict[word] if word in words_dict else 0 for word in words]


def data_generator(index_words, window_size):
    for index, center in enumerate(index_words):
        context_size = random.randint(1, window_size)
        for target in index_words[max(0, index-context_size): index]:
            yield center, target

        for target in index_words[index+1: index + context_size+1]:
            yield center, target


def get_batch(batch_size, iterator):
    while True:
        center_batch = np.zeros(batch_size, dtype='int32')
        target_batch = np.zeros([batch_size, 1])
        for i in range(batch_size):
            center_batch[i], target_batch[i] = next(iterator)
        yield center_batch, target_batch


def process_data(VocabSize, batch_size, window_size):
    file_name = download(FileName)
    words = read_data(file_name)
    words_new =preprocess(words)# words#
    dictionary = build_vocab(words_new, VocabSize)
    index_words = assign_index_to_words(words_new, dictionary)
    itr = data_generator(index_words, window_size)
    return get_batch(batch_size, itr)


def visualise(words_file, word_vector_file):
    words_vector = np.loadtxt(word_vector_file)[:VisualizeWordSize]
    with open(words_file, 'r') as fr:
        words = [line.strip().split() for line in fr.readlines()][:VisualizeWordSize]
    time.sleep(2)
    assert len(words_vector) == len(words)
    print("shape of words tsne/pca", len(words))
    print("shape of word vectors before tsne/pca", words_vector.shape)
    SavePCA(words, words_vector)


def CosineTest(words_file, word_vector_file, querry='south'):
    words_vector = np.loadtxt(word_vector_file)[1:]
    with open(words_file) as fr:
        words = [line.strip().split() for line in fr.readlines()]
    words_and_vectors_dict = {}
    for word, vector in zip(words[1:], words_vector):
        print("word", word, "word shape", len(word))
        print("vectors", word[:10], "vector shape", words_vector.shape)
        words_and_vectors_dict[word[0]] = vector
    del words
    del words_vector
    result = {}
    print("sleeping")
    time.sleep(5)
    print("started")
    print("querry is ", querry)
    for word in words_and_vectors_dict:
        if word != querry:
            try:
                result[word] = cosine(words_and_vectors_dict[querry], words_and_vectors_dict[word])
            except Exception, e:
                print("error", e)
                querry = raw_input()
                result[word] = cosine(words_and_vectors_dict[querry], words_and_vectors_dict[word])

    return sorted(result, key=result.__getitem__, reverse=True)[:50]


def SavePCA(words, vectors):
    pca = PCA(n_components=2)
    vectors_new = pca.fit_transform(vectors)
    print("shape of word vectors after pca", vectors_new.shape)
    # plt.figure(figsize=(200, 200))
    max_x = np.amax(vectors_new[:, 0], axis=0)
    max_y = np.amax(vectors_new[:, 1], axis=0)
    plt.xlim((-max_x, max_x))
    plt.ylim((-max_y, max_y))
    plt.scatter(vectors_new[:, 0], vectors_new[:, 1], 20)

    for word, pos in zip(words, vectors_new):
        plt.annotate(word, (pos[0], pos[1]))
    # plt.savefig('data/pca1.png')
    plt.show()


def SaveTSNE(words, vectors):
    tsne = TSNE(n_components=2)
    tsne_word_vectors = tsne.fit_transform(vectors)
    print("shape of word vectors after pca", tsne_word_vectors.shape)
    # plt.figure(figsize=(200, 200))
    max_x = np.amax(tsne_word_vectors[:, 0], axis=0)
    max_y = np.amax(tsne_word_vectors[:, 1], axis=0)
    plt.xlim((-max_x, max_x))
    plt.ylim((-max_y, max_y))
    plt.scatter(tsne_word_vectors[:, 0], tsne_word_vectors[:, 1], 20)

    for word, pos in zip(words, tsne_word_vectors):
        plt.annotate(word, (pos[0], pos[1]))
    time.sleep(5)
    # SavePCA(words, vectors)
    # plt.savefig('data/tsne.png')
    plt.show()



if __name__ == "__main__":
    # file_name = download(FileName)
    # print("part1 finishes")
    # words = read_data(file_name)[:2000]
    # print("part2 finishes")
    # print("len of words", len(words))
    # words_new = preprocess(words)
    # print("len of new words", len(words_new))
    # del words
    # print("part3 finishes")
    # dicts = build_vocab(words_new, 500)
    # print("part4 finishes")
    # indexed_corpus = assign_index_to_words(words_new, dicts)
    # print("part5 finishes")
    # del words_new
    # itr = data_generator(indexed_corpus, 1)
    # batch_itr = get_batch(10, itr)
    # centers, targets = next(batch_itr)
    # print("len of both", len(centers), len(targets))
    # print(centers, "\n", targets)
    # centers, targets = next(batch_itr)
    # print("len of both", len(centers), len(targets))
    # print(centers, "\n", targets)
    # visualise(VisualizeWordFile, 'data/sample.tsv')
    print(CosineTest(VisualizeWordFile, EmbedVectorFile, querry='summer'))
