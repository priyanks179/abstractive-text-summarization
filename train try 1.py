import os
import util
import numpy as np
import model
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from packages.functions import num_to_var
from torch.autograd import Variable as variable
import random
from collections import Counter
import time

device='cuda'

dir='data/batches/merge_8'
file_list = os.listdir(path=dir)

args={'hidden_size':200,'vocab_size':50000,'embed_size':64,
      'max_enc':400,'max_oovs':400,'batch':2,'max_summ':100}         

enc=model.encoder(args).to(device)
dec=model.attn_decoder(args,enc.vocab_size).to(device)

enc_optimizer=optim.Adam(enc.parameters(),lr=0.001,betas=(0.9, 0.999), eps=1e-08)
dec_optimizer=optim.Adam(dec.parameters(),lr=0.001,betas=(0.9, 0.999), eps=1e-08)
criterion = nn.NLLLoss()


if os.path.isfile('encoder'):
    enc.load_state_dict(torch.load('encoder', map_location=lambda storage, loc: storage))
    dec.load_state_dict(torch.load('decoder', map_location=lambda storage, loc: storage))
    
if os.path.isfile('loss.txt')==0:
    with open('loss.txt', 'w') as output:
        output.write('start')
        output.write('\n')
if os.path.isfile('loss.txt')==1:
    with open('loss.txt', 'r') as output:
        lost_list=output.read().strip().split('\n')    
        lost_list.remove('start')
        for i,data in enumerate(lost_list):
            lost_list[i]=float(data)    


count=0 

if os.path.isfile('count.txt')==1:
    with open('count.txt', 'r') as output:
        count_list=output.read().strip().split('\n')    
        count_list.remove('start')
        for i,data in enumerate(count_list):
            count_list[i]=int(data)   
    count=count_list[-1]
if os.path.isfile('count.txt')==0:
    with open('count.txt', 'w') as output:
        output.write('start')
        output.write('\n')    
iterations,print_every=75000,1 
for iter in range(iterations):
    if count>=len(file_list):
        count=0
    if iter%(8/args['batch'])==0:
        file=file_list[count] 
        path=os.path.join(dir,file)
        input,target,idx2oov=util.get_id(path)
        with open('count.txt', 'a') as output:
            output.write(str(count))
            output.write('\n') 
        count+=1
        ind=0
        
    l=[j for j in range(int(8/args['batch']))]
    inp=num_to_var(input[ind*args['batch']:ind*args['batch']+2,:]).to(device)#8,400
    decoder_inp=num_to_var(target[ind*args['batch']:ind*args['batch']+2,:]).to(device)
    
    enc_optimizer.zero_grad()
    dec_optimizer.zero_grad()
    enc_output,state=enc(inp)#encoder_output=[8,400,512] #state=([2,8,512])*2
    dec_inp=torch.ones(enc_output.size(0),1,dtype=torch.long).to(device)*2#2 for <sos>
    coverage = variable(torch.zeros(dec.batch_size,dec.max_enc)).to(device)#8,400
    dec_state=dec.hid_init(state)
    cov_loss,loss=0,0
    preds=torch.zeros(args['batch'],args['max_summ'])
   
    for i in range(decoder_inp.size(1)):
        dec_state,p_final,coverage,attn,p_copy=dec(enc_output,dec_inp,inp,dec_state,coverage)
        dec_inp=torch.transpose(decoder_inp[:,i].unsqueeze(0),1,0)
        dec_inp=torch.transpose(decoder_inp[:,i].unsqueeze(0),1,0)
        cov_loss+=torch.sum(torch.min(coverage,attn)) 
        loss+=criterion(torch.log(p_final),decoder_inp[:,i])
        pred=torch.argmax(p_final,1)
        for j in range(len(pred)):
            preds[j,i]=pred[j]
    dec_inp.detach()

    if torch.isnan(loss)==0:
        loss.backward() 
        nn.utils.clip_grad_norm_(enc.parameters(),2)
        nn.utils.clip_grad_norm_(dec.parameters(),2)
        enc_optimizer.step()
        dec_optimizer.step()
    else:
        print('loss got nan didn\'t optimized')
    if iter%print_every==0 and iter>0:
        print('file : {} index : {} iteration : {} loss : {} '.format(count-1,ind,iter,loss))
        summ=util.list_to_summ(preds[-1,:].cpu().numpy().tolist(),idx2oov)
        print('predicted summ : ',summ,'\n------------------------------')           
        real_summ=util.list_to_summ(decoder_inp[-1,:].cpu().numpy().tolist(),idx2oov)
        print('real summ : ',real_summ,'\n--------------------------')
        with open('loss.txt', 'a') as output:
            output.write(str((loss).cpu().data.numpy().tolist()))
            output.write('\n')        
    if iter%100 ==0 and torch.isnan(loss)==0:
        
         torch.save(enc.state_dict(),'encoder')
         torch.save(dec.state_dict(),'decoder')  
    with open('param.txt','a') as handle :
        for i in enc.parameters():
            handle.write('enc max param '+str(torch.max(i))+'\n')   
            handle.write('enc min param '+str(torch.min(i))+'\n') 
            handle.write('enc max grad '+str(torch.max(i.grad))+'\n')
            handle.write('enc min grad '+str(torch.min(i.grad))+'\n')   
        for i in dec.parameters():
            handle.write('dec max param '+str(torch.max(i))+'\n')   
            handle.write('dec min param '+str(torch.min(i))+'\n') 
            handle.write('dec max grad '+str(torch.max(i.grad))+'\n')
            handle.write('dec min grad '+str(torch.min(i.grad))+'\n')    
        handle.write('loss '+str(loss)+'\n')  
        handle.write('pfinal min '+str(torch.min(p_final,1))+'\n')  
        handle.write('pfinal max '+str(torch.max(p_final,1))+'\n')   
        handle.write('\n\n')                 
    del inp,decoder_inp,cov_loss,loss
    ind+=1
    
import matplotlib.pyplot as plt
plt.figure()
plt.plot(lost_list)    

for i in dec.parameters():
    print(i)


