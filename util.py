import re
import numpy as np
from packages.vocab import Vocab
import util

vocab=Vocab(50000)
vocab.w2i = np.load('word2idx.npy').item()
vocab.i2w = np.load('idx2word.npy').item()
vocab.count = len(vocab.w2i)

def clean(s):
    s=s.lower().strip()    
#    s = re.sub(r"[^a-zA-Z]+", r" ", s)
    s = re.sub(r"i'm", r"i am ", s)
    s = re.sub(r"he's", r"he is", s)
    s = re.sub(r"she's", r"she is", s)
    s = re.sub(r"you're", r"you are", s)
    s = re.sub(r"we're", r"we are", s)
    s = re.sub(r"they're", r"they are", s)
    s = re.sub(r"i'll", r"i will", s)    
    s = re.sub(r"i'd", r"i would", s)    
    s = re.sub(r"wouldn't", r"would not", s)    
    s = re.sub(r"i've ", r"i have", s) 
    s = re.sub(r"wouldn't", r"would not", s) 
    s = re.sub(r"it s", r"it is", s) 
    s=re.sub(r"\(\)",r"",s)
    s=re.sub(r"\d",r"",s)
    s=re.sub(r"  ",r" ",s)
    s=re.sub(r"[-()\"#/@\^;&<>{}+=~*|%?$!\[\]_©×°«»·`\\]", r"", s)
    s=re.sub(r"['\x80-\xFF]", r"", s)#remove elements not in ascii unicode            
    return(s)
    
def preprocess(s):
    s=s.lower().strip()    
#    s = re.sub(r"[^a-zA-Z]+", r" ", s)
    s = re.sub(r"i'm", r"i am ", s)
    s = re.sub(r"he's", r"he is", s)
    s = re.sub(r"she's", r"she is", s)
    s = re.sub(r"you're", r"you are", s)
    s = re.sub(r"we're", r"we are", s)
    s = re.sub(r"they're", r"they are", s)
    s = re.sub(r"i'll", r"i will", s)    
    s = re.sub(r"i'd", r"i would", s)    
    s = re.sub(r"wouldn't", r"would not", s)    
    s = re.sub(r"i've ", r"i have", s) 
    s = re.sub(r"wouldn't", r"would not", s) 
    s = re.sub(r"it s", r"it is", s) 
    s=re.sub(r"\(\)",r"",s)
    s=re.sub(r"  ",r" ",s)
    s=re.sub(r"[-()\"#/@\^;&<>{}+=~*|%?$!\[\]_©×°«»·`\\]", r"", s)
    s=re.sub(r"['\x80-\xFF]", r"", s)#remove elements not in ascii unicode            
    return(s)    
       
def get_id(file_path,input_len=400,target_len=100,max_oov=400):
    with open(file_path,encoding='utf8') as f:
        text=f.read()
    text=text.split('\n\n')    

    word_list=[]
    art,summ=[],[]
    for t in text:
        temp=t.split(':==:')
        art.append(util.preprocess(temp[0]))
        summ.append(util.preprocess(temp[1]))
        word_list+=util.preprocess(temp[0]).split()
        word_list+=util.preprocess(temp[1]).split()
    del text,t,temp  

    word_list=list(set(word_list))
    oov2idx,idx2oov=vocab.create_oov_list(word_list,max_oov)

    art_max,sum_max=0,0
    for ind,k in enumerate(art):
        if len(k.split())>art_max:
            art_max=len(k.split())
        if  len(summ[ind].split())>sum_max:
            sum_max=len(summ[ind].split())
    if art_max>input_len:
        art_max=input_len
    if sum_max>target_len:
        sum_max=target_len    

    temp=[]
    for index in range(8):
        lst=art[index].split()[:art_max-2]
        lst=vocab.word_list_to_idx_list(lst,oov2idx)
        lst.insert(0,vocab.w2i['<SOS>'])
        lst.insert(len(lst),vocab.w2i['<EOS>'])
        diff=0
        if len(lst)<art_max:
            diff=art_max-len(lst)
            pad=[vocab.w2i['<PAD>']]*diff
            lst=lst+pad
        temp.append(lst)
    inp=np.array(temp).astype(int)

    temp=[]    
    for index in range(8):
        lst=summ[index].split()[:sum_max-1]
        lst=vocab.word_list_to_idx_list(lst,oov2idx)
        lst.insert(len(lst),vocab.w2i['<EOS>'])
        diff=0
        if len(lst)<sum_max:
            diff=sum_max-len(lst)
            pad=[vocab.w2i['<PAD>']]*diff
            lst=lst+pad
        temp.append(lst)
    tar=np.array(temp).astype(int)        
    return(inp,tar,idx2oov)    
    
def list_to_summ(target,idx2oov):
    summ=' '.join(vocab.idx_list_to_word_list(target[:30],idx2oov))
    return(summ)    