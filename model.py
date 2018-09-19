
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


    def hid_init(self,enc_state,device='cuda'):
        #(num_layers * num_directions,batch , hidden_size)
        cell=enc_state[1].view(1,self.batch_size,-1)
        cell=self.hid(cell)
        hid=torch.zeros(1,self.batch_size,self.hidden_size).to(device)
        return((hid,cell))#([1,8,256])*2 and (h0,c0)
        
    def forward(self,enc_output,enc,dec_inp,input,dec_state,coverage,device='cuda'):
        embed=self.embed(dec_inp)#8,1,128
        output,dec_state=self.decoder(embed,dec_state)#output [8,1,256], dec_state [1,8,256]
        
        decoder_state=dec_state[1]#1,8,256
        temp=torch.cat((torch.transpose(output,0,1),decoder_state,coverage.unsqueeze(0)),dim=2)#1,8,512+400
        attn_logit=self.to_enc_vocab(temp)

        attn=torch.softmax(F.relu(attn_logit),2)
        coverage=coverage+attn.squeeze(0)#torch.softmax(F.leaky_relu(self.to_enc_vocab(temp)),1).squeeze(0)
        #[8,1,400]*[8,400,512]-->[8,1,512]
        context=torch.bmm(torch.transpose(attn,0,1),enc_output)#8,1,512

        temp1=torch.cat((context,torch.transpose(dec_state[1],0,1)),dim=2)#8,1,768
        pvocab_attn_logit=self.to_pvocab_attn(temp1.squeeze(1))

        pvocab_attn_logit=torch.cat((pvocab_attn_logit,(torch.ones(self.batch_size,self.max_oovs)*(1e-12)).to(device)),dim=1)
        #pvocab_attn.size() 8,50400
        
        p_vocab=torch.softmax(F.relu(pvocab_attn_logit),1)
        p_vocab=torch.cat((p_vocab,torch.zeros(self.batch_size,self.max_oovs).to(device)),dim=1)
        temp2=torch.cat((context,embed,torch.transpose(dec_state[1],0,1)),dim=2).squeeze(1)#8,896
        p_gen=torch.sigmoid(self.to_pgen(temp2))#8,1

        
        #----------------------------------------------------------------------
        attn_logit=attn_logit.squeeze(0)
        attn=attn.squeeze(0)
        p_final=torch.mul(p_gen,pvocab_attn_logit)
        weighted_attn=torch.mul((1-p_gen),attn_logit)
        p_final=p_final.scatter_add(1, input, weighted_attn)
                

        #----------------------------------------------------------------------


        return(dec_state,p_final,coverage,attn,p_gen)#p_final [8,50400]
        

