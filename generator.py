import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from packages.functions import num_to_var
from torch.autograd import Variable as variable

device='cuda'

def dec_forward(dec,enc_output,dec_inp,inp,dec_state,coverage,input_len,target_len):
#        drop=torch.nn.Dropout(p=0.1, inplace=True)
        input=inp
        b, t_k, n=list(enc_output.size())
        
        embed=dec.embed(dec_inp)#8,1,128
        output,dec_state=dec.decoder(embed,dec_state)#output [8,1,256], dec_state [1,8,256]
        
        enc_features=dec.encoder_output_proj(enc_output.contiguous().view(-1,dec.hidden_size*2))#8*400,512
#        enc_features=drop(enc_features)
        state_combined=torch.cat((dec_state[0],dec_state[1]),2)#1,8,512
        dec_features=dec.decoder_state_proj(state_combined.view(-1,dec.hidden_size*2))#1*8,512
#        dec_features=drop(dec_features)
        dec_fea_expanded = dec_features.unsqueeze(1).expand(b, t_k, n).contiguous() # 8,400,512
        dec_fea_expanded = dec_fea_expanded.view(-1, n)  # 8*400,512
        coverage_input=coverage.view(-1,1)#8*400,1
        coverage_feature = dec.coverage_proj(coverage_input)  # 8*400 , 512
        attn_logit=coverage_feature+dec_fea_expanded+enc_features# 8*400 , 512
        attn_logit = torch.tanh(attn_logit)
        attn_logit=dec.encoder_vocab_proj(attn_logit).view(-1,t_k)# B x t_k
        enc_padding_mask=(input!=0).to(device).to(torch.float)
        attn_logit=attn_logit.squeeze(0)[:,:input_len]
        attn_logit=attn_logit.squeeze(1)
        attn = F.softmax(attn_logit, dim=1)*enc_padding_mask[:,:input_len]

        attn=attn.unsqueeze(1)#b,1,tk

        extra=torch.zeros((dec.batch_size,dec.max_enc-input_len),dtype=torch.float).to(device)
        attn=torch.cat((attn.squeeze(1),extra),1)
        coverage=coverage+attn#torch.softmax(F.leaky_relu(self.to_enc_vocab(temp)),1).squeeze(0)
        #[8,1,400]*[8,400,512]-->[8,1,512]
        attn=attn.unsqueeze(1)
        context=torch.bmm(attn,enc_output)#8,1,512

        temp1=torch.cat((context,output),dim=2)#8,1,768
#        temp1=drop(temp1)
        temp1=dec.out1(temp1.squeeze(1))
        pvocab_attn_logit=dec.to_pvocab_attn(temp1.squeeze(1))#8,50k

#        pvocab_attn_logit=torch.cat((pvocab_attn_logit,(torch.ones(self.batch_size,self.max_oovs)*(1e-12)).to(device)),dim=1)
        #pvocab_attn.size() 8,50400
        del temp1
        p_vocab=torch.softmax(pvocab_attn_logit,1)
        p_vocab=torch.cat((p_vocab,torch.zeros(dec.batch_size,dec.max_oovs).to(device)),dim=1)#8,50400
        
        temp2=torch.cat((context,embed,torch.transpose(dec_state[1],0,1),torch.transpose(dec_state[0],0,1)),dim=2).squeeze(1)#8,896
#        temp2=drop(temp2)
        p_gen=torch.sigmoid(dec.to_pgen(temp2))#8,1
        del embed,context,temp2
        #######################################################################
        attn_logit=attn.squeeze(1)[:,:input_len]

        pvocab_attn_logit=p_vocab
        attn=attn.squeeze(0)
        p_final=torch.mul(p_gen,pvocab_attn_logit)
        weighted_attn=torch.mul((1-p_gen),attn_logit)
        p_final=p_final.scatter_add(1, input[:,:input_len], weighted_attn) 
        #######################################################################
        temp=(p_final==0).to(torch.float)
        temp=temp*torch.tensor(1e-12).to(device)       
        p_final+=temp        
        return(dec_state,p_final,coverage,attn,p_gen)#p_final [8,50400]      
        
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
        
    def forward(self,unk_inp,device='cuda'):

        enc_embed=self.embedding(unk_inp)#8,400,128
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
        
        self.encoder_output_proj=nn.Linear(self.hidden_size*2,self.hidden_size*2,bias=False)
        self.decoder_state_proj=nn.Linear(self.hidden_size*2,self.hidden_size*2,bias=False)
        self.coverage_proj=nn.Linear(1,self.hidden_size*2,bias=False)
        self.encoder_vocab_proj=nn.Linear(self.hidden_size*2,1,bias=False)


        self.embed=nn.Embedding(vocab_size+self.max_oovs,embed_size)
        self.decoder=nn.LSTM(input_size=embed_size,hidden_size=hidden_size,
                             batch_first=True)    
        self.hid=nn.Linear(2*hidden_size,hidden_size)
        self.cell=nn.Linear(2*hidden_size,hidden_size)
#        self.to_enc_vocab=nn.Linear(2*hidden_size+self.max_enc,self.max_enc)
        self.to_pvocab_attn=nn.Linear(hidden_size,self.vocab_size)
        self.to_pgen=nn.Linear(4*hidden_size+embed_size,1)
        self.out1=nn.Linear(3*hidden_size,hidden_size)

    def hid_init(self,enc_state,device='cuda'):
        #(num_layers * num_directions,batch , hidden_size)
        cell=enc_state[1].view(1,self.batch_size,-1)
        cell=F.relu(self.cell(cell))
        hidden=enc_state[0].view(1,self.batch_size,-1)
        hidden=F.relu(self.hid(hidden))
#        hid=torch.zeros(1,self.batch_size,self.hidden_size).to(device)
        return((hidden,cell))#([1,8,256])*2 and (h0,c0)        

def genrate(enc,dec,inp,decoder_inp,args,input_len,target_len):
    enc_output,state=enc(inp)#encoder_output=[8,400,512] #state=([2,8,512])*2
    dec_inp=torch.ones(enc_output.size(0),1,dtype=torch.long).to(device)*2#2 for <sos>
    coverage = variable(torch.zeros(dec.batch_size,dec.max_enc)).to(device)#8,400
    dec_state=dec.hid_init(state)
    cov_loss=0
    preds_summ=variable(torch.zeros(args['batch'],decoder_inp.size(1)),requires_grad=True)
    for i in range(decoder_inp.size(1)):
        dec_state,p_final,coverage,attn,p_gen=dec_forward(dec,enc_output,dec_inp,inp,dec_state,coverage,input_len,target_len)
        dec_inp=torch.transpose(decoder_inp[:,i].unsqueeze(0),1,0)
        cov_loss+=torch.sum(torch.min(coverage,attn)) 
        p_final=torch.softmax(p_final,1)
#        p_final+=((p_final==0)*1e-20).to(torch.float)
        pred=torch.argmax(p_final,1)
        for j in range(len(pred)):
            preds_summ[j,i]=pred[j]
        if i==0:
            preds=p_final.unsqueeze(1)
        else:
            preds=torch.cat((preds,p_final.unsqueeze(1)),1)
    return(preds_summ,preds,p_final,attn)
    