from collections import Counter
import pickle
import spacy

class Vocab(object):
    def __init__(self,max_size):
        self.w2i = {}
        self.i2w = {}
        self.count = 0
        self.max_size = max_size
        self.nlp = spacy.load('en')
        self.counter = Counter()
        
        for w in ['<PAD>','<UNK>','<SOS>','<EOS>']:
            self.w2i[w] = self.count
            self.i2w[self.count] = w
            self.count +=1
            
    def word2idx(self, word):
        if word not in self.w2i:
            return self.w2i['<UNK>']
        return self.w2i[word]
            
    def idx2word(self, idx): 
        if idx not in self.i2w:
            return '<UNK>'
        return self.i2w[idx]
    
    def create_oov_list(self, word_list, max_oovs):
        oov2idx = {}  
        idx2oov = {}
        oov_count=0
        for word in word_list:
            if (word not in oov2idx) & (word not in self.w2i):
                oov2idx[word] = self.count + oov_count
                idx2oov[self.count+oov_count] = word
                oov_count+=1
            if oov_count>=max_oovs: 
                return oov2idx, idx2oov
        return oov2idx,idx2oov

    def idx_list_to_word_list(self, idx_list, idx2oov={}):
        if type(idx_list[0])!=int:    
            idx_list = [int(idx) for idx in idx_list]
        out = []    
        for idx in idx_list: 
            if idx in idx2oov:
                out.append(idx2oov[idx])
            else:          
                out.append(self.idx2word(idx))
        return out
                
    def word_list_to_idx_list(self, word_list, oov2idx={}):
        out = []       
        for word in word_list:
            if word in oov2idx:
                out.append(oov2idx[word])
            else:
                out.append(self.word2idx(word))
        return out
    
    def add_to_vocab(self, word_list):
        for word in word_list:
            if self.count>=self.max_size:
                print(" Vocabulary max size reached!")
                return 
            else:
                if (word not in self.w2i) & (word not in ['<PAD>','<S>','</S>','<UNK>']):
                    self.w2i[word]=self.count
                    self.i2w[self.count]=word
                    self.count+=1

    def tokenize(self, text):
        tokenized = self.nlp(text)
        out = [x.text for x in tokenized]
        return out
    
    def feed_to_counter(self, word_list):
        counter = Counter(word_list)
        self.counter = self.counter + counter
        return counter
    
    def preprocess_string(self, text, preprocess_list):
        for tup in preprocess_list:
            from_str, to_str = tup
            text = text.replace(from_str, to_str)
        return text
    
    def counter_to_vocab(self, counter):
        word_list = [tup[0] for tup in counter.most_common()[:self.max_size]]
        rem=list(self.nlp(' '.join(word_list)).ents)
        for i,word in enumerate(rem):
            rem[i]=str(word)
        word_list=list(set(word_list)-set(rem))
        self.add_to_vocab(word_list)
        return(word_list)

          
        
    
                
            
            