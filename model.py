
import numpy as np
import random
import os
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from packages.functions import num_to_var
from torch.autograd import Variable as variable
import time
from collections import Counter
device='cuda'


    
class encoder(nn.Module):
    def __init__(self,args):
        super(encoder, self).__init__()
        hidden_size = args['hidden_size']
        vocab_size = args['vocab_size']
        embed_size = args['embed_size']
    
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.max_enc=args['max_enc']
        self.max_oovs = args['max_oovs']
        self.batch_size=args['batch']
        self.vocab_size=vocab_size+self.max_oovs#to genrate source vocablury
        
        self.embedding=nn.Embedding(self.vocab_size,embed_size)
        self.lstm=nn.LSTM(input_size=embed_size,hidden_size=hidden_size,
                             batch_first=True,bidirectional=True)  
        
    def forward(self,input):
        enc_embed=self.embedding(input)#8,400,128
        enc_output,state=self.lstm(enc_embed)  
        if enc_output.size(1)<self.max_enc:
            diff=self.max_enc-enc_output.size(1)
            temp=torch.zeros(self.batch_size,diff,enc_output.size(2)).to(device)
            enc_output=torch.cat((enc_output,temp),dim=1)
        return(enc_output,state)#encoder_output=[8,400,512] #state=([2,8,512])*2 (h0,c0)           
 


class attn_decoder(nn.Module):
    def __init__(self,args,encoder_vocab):
        super(attn_decoder, self).__init__()
        hidden_size = args['hidden_size']
        vocab_size = args['vocab_size']
        embed_size = args['embed_size']
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.vocab_size=vocab_size
        self.max_enc=args['max_enc']
        self.max_oovs = args['max_oovs']        
        self.batch_size=args['batch']
        self.encoder_vocab=encoder_vocab

        self.embed=nn.Embedding(vocab_size+self.max_oovs,embed_size)
        self.decoder=nn.LSTM(input_size=embed_size,hidden_size=hidden_size,
                             batch_first=True)    
        self.hid=nn.Linear(2*hidden_size,hidden_size)
        self.to_enc_vocab=nn.Linear(2*hidden_size+self.max_enc,self.max_enc)
        self.to_pvocab_attn=nn.Linear(hidden_size*3,self.vocab_size)
        self.to_pgen=nn.Linear(3*hidden_size+embed_size,1)

    def hid_init(self,enc_state):
        #(num_layers * num_directions,batch , hidden_size)
        cell=enc_state[1].view(1,self.batch_size,-1)
        cell=self.hid(cell)
        hid=torch.zeros(1,self.batch_size,self.hidden_size).to(device)
        return((hid,cell))#([1,8,256])*2 and (h0,c0)
        
    def forward(self,enc_output,dec_inp,input,dec_state,coverage):
        embed=self.embed(dec_inp)#8,1,128
        output,dec_state=self.decoder(embed,dec_state)#output [8,1,256], dec_state [1,8,256]
        decoder_state=dec_state[1]#1,8,256
        temp=torch.cat((torch.transpose(output,0,1),decoder_state,coverage.unsqueeze(0)),dim=2)#1,8,512+400
        attn=F.softmax(F.relu(self.to_enc_vocab(temp)),dim=2)#1,8,400
        coverage=coverage+attn.squeeze(0)
        #[8,1,400]*[8,400,512]-->[8,1,512]
        context=torch.bmm(torch.transpose(attn,0,1),enc_output)#8,1,512

        temp1=torch.cat((context,torch.transpose(dec_state[1],0,1)),dim=2)#8,1,768
        pvocab_attn=self.to_pvocab_attn(temp1.squeeze(1))
        pvocab_attn=torch.softmax(pvocab_attn,dim=1)#8,50k
        pvocab_attn=torch.cat((pvocab_attn,torch.ones(self.batch_size,self.max_oovs).to(device)*(1e-12)),dim=1)
        #pvocab_attn.size() 8,50400

        temp2=torch.cat((context,embed,torch.transpose(dec_state[1],0,1)),dim=2).squeeze(1)#8,896
        p_gen=torch.sigmoid(self.to_pgen(temp2))#8,1
        
        #----------------------------------------------------------------------
        inp=input.cpu().numpy()
        attn=attn.squeeze(0)
        b,in_seq=self.batch_size,self.max_enc
        numbers = inp.reshape(-1).tolist()#get numbers in input
        set_numbers = list(set(numbers))
        if 0 in set_numbers:
            set_numbers.remove(0)
        c = Counter(numbers)
        dup_list = [k for k in set_numbers if (c[k]>1)]#list of number>1
        masked_idx_sum = np.zeros([self.batch_size,self.max_enc],dtype=float)#8,400
        #dup_attn_sum 8,400
        dup_attn_sum = variable(torch.FloatTensor(np.zeros([self.batch_size,self.max_enc],dtype=float))).to(device)

        for dup in dup_list:
            mask = np.array(inp==dup, dtype=float)#put 1 where no occured in mask (8,400)
            masked_idx_sum += mask
            # 8,400 * 8,400 --> 8,400
            attn_mask = torch.mul(variable(torch.Tensor(mask)).to(device),attn.squeeze(0))#put attn where those occured    
            attn_sum = attn_mask.sum(1).unsqueeze(1)#8,1
            dup_attn_sum += torch.mul(attn_mask,attn_sum)#8,1 * 8,400 -->8,400

        masked_idx_sum = variable(torch.Tensor(masked_idx_sum)).to(device)    

        attn = torch.mul(attn,(1-masked_idx_sum))+dup_attn_sum#8,400

        batch_indices = torch.arange(start=0, end=self.batch_size).long()
        batch_indices = batch_indices.expand(self.max_enc, self.batch_size).transpose(1, 0).contiguous().view(-1)#3200
        idx_repeat = torch.arange(start=0, end=in_seq).repeat(b).long()#3200

        p_copy = variable(torch.zeros(b,self.vocab_size+self.max_oovs)).to(device)#8,50200
        word_indices = input.reshape(-1)
        p_copy[batch_indices,word_indices] += attn[batch_indices,idx_repeat]
        
        
        #----------------------------------------------------------------------

        p_final=torch.mul(p_gen,pvocab_attn)+torch.mul((1-p_gen),p_copy)
#        del temp1,temp2,temp,numbers,unq_num,num
        return(dec_state,p_final,coverage,attn,p_copy)#p_final [8,50400]
        
##torch.transpose(decoder_inp[:,0].unsqueeze(0),1,0)           
#args={'hidden_size':256,'vocab_size':50000,'embed_size':128,'max_enc':400,'max_oovs':400,'batch':8}         
#enc=encoder(args).to(device)
#dec=attn_decoder(args,enc.vocab_size).to(device) 
#
#
#decoder_inp=num_to_var(target)#8,92
#enc_output,state=enc(inp)#encoder_output=[8,400,512] #state=([2,8,512])*2           
#        
#dec_inp=torch.ones(enc_output.size(0),1,dtype=torch.long).to(device)*2#2 for <sos>
#coverage = variable(torch.zeros(dec.batch_size,dec.max_enc)).to(device)#8,400
#dec_state=dec.hid_init(state)
#cov_loss=0
#for i in range(decoder_inp.size(1)):
#    dec_state,p_final,coverage,attn=dec(dec_inp,dec_state,coverage)
#    dec_inp=torch.transpose(decoder_inp[:,i].unsqueeze(0),1,0)
#    cov_loss=torch.sum(torch.min(coverage,attn))           



