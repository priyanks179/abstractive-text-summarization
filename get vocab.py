import collections
from packages.vocab import Vocab
import os
import pickle
import spacy
from collections import Counter
import util
from random import shuffle
from packages.vocab import Vocab
import os

nlp=spacy.load('en')
input_dir = 'data/batches/cnn_8'


vocab=Vocab(50000)
   

batch_count=0
file_list = os.listdir(path=input_dir)
shuffle(file_list)

for file in file_list[223:]: 
    dir=os.path.join(input_dir,file)
    with open(dir,encoding="utf8") as f:
        text=f.read()
        text = vocab.preprocess_string(text,[(":==:"," "),('\n\n', " ")])
        text=util.clean(text.strip())
        word_list = vocab.tokenize(text)
    counter=vocab.feed_to_counter(word_list)
    a=vocab.counter_to_vocab(counter) 
    print(file,' completed')    
      
with open('counter_cnn.pckl','wb') as f:
    pickle.dump(vocab.counter,f)
      
import numpy as np
np.save('word2idx.npy',vocab.w2i)
np.save('idx2word.npy',vocab.i2w)
  





