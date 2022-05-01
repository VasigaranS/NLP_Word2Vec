import os
import time
import argparse
import string
from collections import defaultdict
import numpy as np
from sklearn.neighbors import NearestNeighbors
import random
from tqdm import tqdm


"""
This file is part of the computer assignments for the course DD2418 Language engineering at KTH.
Created 2020 by Dmytro Kalpakchi.
"""


class Word2Vec(object):
    def __init__(self, filenames, dimension=200, window_size=3, nsample=10,
                 learning_rate=0.025, epochs=5, use_corrected=True, use_lr_scheduling=True):
        """
        Constructs a new instance.
        
        :param      filenames:      A list of filenames to be used as the training material
        :param      dimension:      The dimensionality of the word embeddings
        :param      window_size:    The size of the context window
        :param      nsample:        The number of negative samples to be chosen
        :param      learning_rate:  The learning rate
        :param      epochs:         A number of epochs
        :param      use_corrected:  An indicator of whether a corrected unigram distribution should be used
        """
        self.__pad_word = '<pad>'
        self.__sources = filenames
        self.__H = dimension
        self.__lws = window_size
        self.__rws = window_size
        self.__C = self.__lws + self.__rws
        self.__init_lr = learning_rate
        self.__lr = learning_rate
        self.__nsample = nsample
        self.__epochs = epochs
        self.__nbrs = None
        self.__use_corrected = use_corrected
        self.__use_lr_scheduling = use_lr_scheduling
        self.unigram_prob=dict()
        self.corrected_unigram_prob=dict()
        self.word_count=dict()
        self.tokens_in_corpus=0


    def init_params(self, W, w2i, i2w):
        self.__W = W
        #self.__U = W
        self.__w2i = w2i
        self.__i2w = i2w


    @property
    def vocab_size(self):
        return self.__V
        

    def clean_line(self, line):
        """
        The function takes a line from the text file as a string,
        removes all the punctuation and digits from it and returns
        all words in the cleaned line as a list
        
        :param      line:  The line
        :type       line:  str
        """
        line=line.replace(string.punctuation,'')
        line = line.translate(str.maketrans('', '', string.punctuation))
        line = line.translate(str.maketrans('', '', string.digits))
        #CLEprint(line)
        return line.split()


    def text_gen(self):
        """
        A generator function providing one cleaned line at a time

        This function reads every file from the source files line by
        line and returns a special kind of iterator, called
        generator, returning one cleaned line a time.

        If you are unfamiliar with Python's generators, please read
        more following these links:
        - https://docs.python.org/3/howto/functional.html#generators
        - https://wiki.python.org/moin/Generators
        """
        for fname in self.__sources:
            with open(fname, encoding='utf8', errors='ignore') as f:
                for line in f:
                    yield self.clean_line(line)


    def get_context(self, sent, i):
        """
        Returns the context of the word `sent[i]` as a list of word indices
        
        :param      sent:  The sentence
        :type       sent:  list
        :param      i:     Index of the focus word in the sentence
        :type       i:     int
        """
        #
        # REPLACE WITH YOUR CODE
        # 
        res=list()
        for j in range(1,self.__rws+1):
                        if(i+j)<len(sent):
                            #self.__cv[word]=self.__cv.get(word,np.zeros(self.__dim))  
                            res.append(self.__w2i[sent[j]])
                                      
        for j in range(1, self.__lws+1):
                        if(i-j)>0:
                            res.append(self.__w2i[sent[j]])
                            



        return res


    def skipgram_data(self):
        """
        A function preparing data for a skipgram word2vec model in 3 stages:
        1) Build the maps between words and indexes and vice versa
        2) Calculate the unigram distribution and corrected unigram distribution
           (the latter according to Mikolov's article)
        3) Return a tuple containing two lists:
            a) list of focus words
            b) list of respective context words
        """
        #
        # REPLACE WITH YOUR CODE
        # 

        self.__w2i =dict()
        self.__i2w =dict()
        i=0
        for line in self.text_gen():
            for word in line:
                self.tokens_in_corpus+=1
                self.word_count[word]=self.word_count.get(word,0)+1
                if(word not in self.__w2i):
                    self.__w2i[word]=i
                    self.__i2w[i]=word
                    i+=1
        den=0


        for word in self.word_count:
            self.unigram_prob[word]=self.word_count[word]/self.tokens_in_corpus
            den=den+pow(self.unigram_prob[word],0.75)

        for word in self.word_count:
            self.corrected_unigram_prob[word]=pow(self.unigram_prob[word],0.75)/den

        focus=[]
        context=[]

        for line in self.text_gen():
            for f in range(len(line)):
                focus.append(self.__w2i[line[f]])
                context.append(self.get_context(line,f)) #list of list





        return focus, context


    def sigmoid(self, x):
        """
        Computes a sigmoid function
        """
        return 1 / (1 + np.exp(-x))


    def negative_sampling(self, number, xb, pos):
        """
        Sample a `number` of negatives examples with the words in `xb` and `pos` words being
        in the taboo list, i.e. those should be replaced if sampled.
        
        :param      number:     The number of negative examples to be sampled
        :type       number:     int
        :param      xb:         The index of the current focus word
        :type       xb:         int
        :param      pos:        The index of the current positive example
        :type       pos:        int
        """
        #
        # REPLACE WITH YOUR CODE
        #
        negative_indexes=list()

        i=0
        while(i<number):

            if(self.__use_corrected==True):
                rand_neg_sample=random.choices(population=list(self.corrected_unigram_prob.keys()),weights=list(self.corrected_unigram_prob.values()),k=1)[0]
            else:
                rand_neg_sample=random.choices(population=list(self.unigram_prob.keys()),weights=list(self.unigram_prob.values()),k=1)[0]



            
            rand_neg_index=self.__w2i[rand_neg_sample]
            if(rand_neg_index!=xb and rand_neg_index!=pos and rand_neg_index not in negative_indexes):
                negative_indexes.append(rand_neg_index)
                i+=1

        return negative_indexes


    def train(self):
        """
        Performs the training of the word2vec skip-gram model
        """
        x, t = self.skipgram_data()
        N = len(x)
        self.__V=N
        print("Dataset contains {} datapoints".format(N))

        # REPLACE WITH YOUR RANDOM INITIALIZATION
        self.__W = np.random.uniform(size = (N, self.__H))#focus vectore
        self.__U = np.random.uniform(size = (N, self.__H))#context vector
        alpha_start=self.__lr

        sigmoid_vectorize = np.vectorize(self.sigmoid) 

        lr=self.__lr

        for ep in range(self.__epochs):
            for i in tqdm(range(N)):
                #
                # YOUR CODE HERE 
                #
                if(self.__use_lr_scheduling==True):


                    if(lr < self.__lr *0.0001):
                        lr=self.__lr*0.0001
                    else:
                        lr=self.__lr*(1-(ep*N+i)/(self.__epochs*N+1))

                else:
                    lr=self.__lr


                for u in t[i]:
                    p_sigmoid=sigmoid_vectorize(self.__U[u].T @ self.__W[i])
                    self.__W[i]=self.__W[i]-lr*(self.__U[u].dot(p_sigmoid-1))

                    self.__U[u]=self.__U[u]-lr*(self.__W[i].dot(p_sigmoid-1))

                    negative_samples=self.negative_sampling(self.__nsample, i, u)

                    for n in negative_samples:
                        n_sigmoid=sigmoid_vectorize(self.__U[n].T @self.__W[i])
                        self.__W[i]=self.__W[i]-lr*(self.__U[n].dot(n_sigmoid))

                        self.__U[n]=self.__U[n]-lr*(self.__W[i].dot(n_sigmoid))



        pass


    def find_nearest(self, words, metric):
        """
        Function returning k nearest neighbors with distances for each word in `words`
        
        We suggest using nearest neighbors implementation from scikit-learn 
        (https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html). Check
        carefully their documentation regarding the parameters passed to the algorithm.
    
        To describe how the function operates, imagine you want to find 5 nearest neighbors for the words
        "Harry" and "Potter" using some distance metric `m`. 
        For that you would need to call `self.find_nearest(["Harry", "Potter"], k=5, metric='m')`.
        The output of the function would then be the following list of lists of tuples (LLT)
        (all words and distances are just example values):
    
        [[('Harry', 0.0), ('Hagrid', 0.07), ('Snape', 0.08), ('Dumbledore', 0.08), ('Hermione', 0.09)],
         [('Potter', 0.0), ('quickly', 0.21), ('asked', 0.22), ('lied', 0.23), ('okay', 0.24)]]
        
        The i-th element of the LLT would correspond to k nearest neighbors for the i-th word in the `words`
        list, provided as an argument. Each tuple contains a word and a similarity/distance metric.
        The tuples are sorted either by descending similarity or by ascending distance.
        
        :param      words:   Words for the nearest neighbors to be found
        :type       words:   list
        :param      metric:  The similarity/distance metric
        :type       metric:  string
        """
        #
        # REPLACE WITH YOUR CODE
        #

        neigh=NearestNeighbors(n_neighbors=5, metric=metric)
        neigh.fit(self.__W) #change

        list_ofclosed_words=list()


        for word in words:

            dist_list, indexes=neigh.kneighbors([self.__W[self.__w2i[word]]],return_distance=True)#change
            close_words=list()

            for i in range(len(indexes[0])):
                index=indexes[0][i]
                dist=dist_list[0][i]
                print(index)
                print('here')
                print(self.__i2w)
                w=self.__i2w[index]
                
                #w=index
                close_words.append((w,dist))
            list_ofclosed_words.append(close_words)

        return list_ofclosed_words


    def write_to_file(self):
        """
        Write the model to a file `w2v.txt`
        """
        try:
            with open("w2v.txt", 'w') as f:
                W = self.__W
                f.write("{} {}\n".format(self.__V, self.__H))
                for i, w in enumerate(self.__i2w):
                    f.write(str(self.__i2w[w]) + " " + " ".join(map(lambda x: "{0:.6f}".format(x), W[i,:])) + "\n")
        except:
            print("Error: failing to write model to the file")


    @classmethod
    def load(cls, fname):
        """
        Load the word2vec model from a file `fname`
        """
        w2v = None
        try:
            with open(fname, 'r') as f:
                V, H = (int(a) for a in next(f).split())
                w2v = cls([], dimension=H)

                W, i2w, w2i = np.zeros((V, H)), [], {}
                for i, line in enumerate(f):
                    parts = line.split()
                    word = parts[0].strip()
                    w2i[word] = i
                    W[i] = list(map(float, parts[1:]))
                    i2w.append(word)

                w2v.init_params(W, w2i, i2w)
        except:
            print("Error: failing to load the model to the file")
        return w2v


    def interact(self):
        """
        Interactive mode allowing a user to enter a number of space-separated words and
        get nearest 5 nearest neighbors for every word in the vector space
        """
        print("PRESS q FOR EXIT")
        text = input('> ')
        while text != 'q':
            text = text.split()
            neighbors = self.find_nearest(text, 'cosine')

            for w, n in zip(text, neighbors):
                print("Neighbors for {}: {}".format(w, n))
            text = input('> ')


    def train_and_persist(self):
        """
        Main function call to train word embeddings and being able to input
        example words interactively
        """
        self.train()
        self.write_to_file()
        self.interact()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='word2vec embeddings toolkit')
    parser.add_argument('-t', '--text', default='harry_potter_1.txt',
                        help='Comma-separated source text files to be trained on')
    parser.add_argument('-s', '--save', default='w2v.txt', help='Filename where word vectors are saved')
    parser.add_argument('-d', '--dimension', default=200, help='Dimensionality of word vectors')
    parser.add_argument('-ws', '--window-size', default=2, help='Context window size')
    parser.add_argument('-neg', '--negative_sample', default=10, help='Number of negative samples')
    parser.add_argument('-lr', '--learning-rate', default=0.025, help='Initial learning rate')
    parser.add_argument('-e', '--epochs', default=1, help='Number of epochs')
    parser.add_argument('-uc', '--use-corrected', action='store_true', default=True,
                        help="""An indicator of whether to use a corrected unigram distribution
                                for negative sampling""")
    parser.add_argument('-ulrs', '--use-learning-rate-scheduling', action='store_true', default=True,
                        help="An indicator of whether using the learning rate scheduling")
    args = parser.parse_args()

    if os.path.exists(args.save):
        w2v = Word2Vec.load(args.save)
        if w2v:
            w2v.interact()
    else:
        w2v = Word2Vec(
            args.text.split(','), dimension=args.dimension, window_size=args.window_size,
            nsample=args.negative_sample, learning_rate=args.learning_rate, epochs=args.epochs,
            use_corrected=args.use_corrected, use_lr_scheduling=args.use_learning_rate_scheduling
        )
        w2v.train_and_persist()
