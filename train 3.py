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
from keras.utils import to_categorical


device='cuda'
dir='pointer nw 2/data/batches/merge_8'
file_list = os.listdir(path=dir)

input_len,target_len=100,50

args={'hidden_size':200,'vocab_size':50000,'embed_size':64,
      'max_enc':400,'max_oovs':400,'batch':2,'max_summ':100} 

enc=model.encoder(args).to(device)
dec=model.attn_decoder(args,enc.vocab_size).to(device)
discrim=model.CnnTextClassifier(args['vocab_size']+args['max_oovs']).to(device)

#enc_optimizer=optim.Adam(enc.parameters(),lr=0.001 ,betas=(0.9, 0.999), eps=1e-08)
#dec_optimizer=optim.Adam(dec.parameters(),lr=0.001 ,betas=(0.9, 0.999), eps=1e-08)
#discrim_optimizer = optim.Adam(discrim.parameters(), lr=0.001,betas=(0.9, 0.999), eps=1e-08)
enc_optimizer=optim.Adagrad(enc.parameters(),lr=0.15 ,initial_accumulator_value=0.1)
dec_optimizer=optim.Adagrad(dec.parameters(),lr=0.15 ,initial_accumulator_value=0.1)
discrim_optimizer=optim.Adagrad(discrim.parameters(),lr=0.15 ,initial_accumulator_value=0.1)

criterion = nn.NLLLoss()
discrim_criterion = nn.BCELoss()

if os.path.isfile('encoder'):
    enc.load_state_dict(torch.load('encoder', map_location=lambda storage, loc: storage))
    dec.load_state_dict(torch.load('decoder', map_location=lambda storage, loc: storage))




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
        

def ones_target(size,device):
    data = variable(torch.ones(size, 1).to(device))#give matrix of size*1
    return data

def zeros_target(size,device):
    data = variable(torch.zeros(size, 1).to(device))
    return data


def train_discrim(optimizer,real_data,fake_data):
    N=real_data.size(0)#give batch size
    optimizer.zero_grad()#reset grad to 0 
    #1.1 train on real data
    real_data=real_data.cpu().numpy()
    real_data=to_categorical(real_data,num_classes=args['vocab_size']+args['max_oovs'])
    real_data=torch.tensor(real_data).to(device)
    predic_real=discrim(real_data)
    error_real=discrim_criterion(predic_real,ones_target(N,device))
    error_real.backward()#got grad but didnot reset weights
    #1.2 train on fake data
    predic_fake=discrim(fake_data)
    error_fake=discrim_criterion(predic_fake,zeros_target(N,device))
    error_fake.backward()
    #1.3 update weight with grad
    optimizer.step()
    error_fake,error_real=error_fake.detach(),error_real.detach()
    #return error and predic
    return error_real+error_fake,predic_real,predic_fake

    
def train_genrator(enc_optimizer,dec_optimizer,fake_data):
    N = fake_data.size(0)
    # Reset gradients
    enc_optimizer.zero_grad()
    dec_optimizer.zero_grad()
    discrim_optimizer.zero_grad()
    # Sample noise and generate fake data
    prediction = discrim(fake_data)
    # Calculate error and backpropagate
    error = discrim_criterion(prediction, ones_target(N,device))
    error.backward()
    # Update weights with gradients
#    nn.utils.clip_grad_norm_(dec.parameters(),2)
#    nn.utils.clip_grad_norm_(enc.parameters(),2)
    enc_optimizer.step()
    dec_optimizer.step()
    # Return error
    return error

iterations,print_every=75000,1

#iterations,count=1,0
for iter in range(iterations):
    cov_loss,loss=0,0
    if count>=len(file_list):
        count=0
    if iter%(8/args['batch'])==0:
        file=file_list[count] 
        path=os.path.join(dir,file)
        input,target,idx2oov=util.get_id(path,input_len,target_len)
        with open('count.txt', 'a') as output:
            output.write(str(count))    
            output.write('\n') 
        count+=1
        ind=0

    l=[j for j in range(int(8/args['batch']))]
    inp=num_to_var(input[ind*args['batch']:ind*args['batch']+args['batch'],:]).to(device)#8,400
    decoder_inp=num_to_var(target[ind*args['batch']:ind*args['batch']+args['batch'],:]).to(device)
    
    preds_summ,preds,p_final,attn=model.genrate(enc,dec,inp,decoder_inp,args,input_len,target_len)
    
    #train discrim
#    preds_for_discrim=preds.detach()
#    d_error,d_pred_real,d_pred_fake=\
#            train_discrim(discrim_optimizer,decoder_inp,preds_for_discrim)#predic_fake is discim redic on pred from gen

#    print(d_error.cpu().numpy(),' ','d_error')
#    d_error=d_error.detach()
#    del d_pred_real,d_pred_fake
    
    #train genrator
#    preds_summ,preds,p_final,attn=model.genrate(enc,dec,inp,decoder_inp,args,input_len,target_len)
    g_error=train_genrator(enc_optimizer,dec_optimizer,preds) 
    
    cnt,sum=0,0
    for i in dec.parameters():
        sum+=torch.max(i.grad)
        cnt+=1    

    if(iter%print_every==0):
        print('iter: ',iter,' g_error: ',g_error.detach().cpu().numpy(),' max dec grad: ',(sum/cnt).cpu().tolist())
        summ=util.list_to_summ(preds_summ[-1,:].detach().cpu().numpy().tolist(),idx2oov)
        real_summ=util.list_to_summ(decoder_inp[-1,:].cpu().numpy().tolist(),idx2oov)
        print('------------------------------')
        print('pred summ : ',summ,'\n------------------------------')    
        print('real summ : ',real_summ,'\n--------------------------')
        print('p_final max: ',p_final.max(1))
#        with open('discrim_loss.txt','a') as handle:
#            handle.write(str(d_error.cpu().numpy())+'\n')
        with open('gen_loss.txt','a') as handle:
            handle.write(str(g_error.detach().tolist())+'\n')
        with open('count.txt','a') as handle:
            handle.write(str(count)+'\n')
    if (iter%1000==0):
        torch.save(enc.state_dict(),'encoder')
        torch.save(dec.state_dict(),'decoder')
        torch.save(discrim.state_dict(),'discrim')
            
    del preds_summ,preds,p_final,attn,g_error
            


#fake_data=preds
import matplotlib.pyplot as plt
plt.plot(lost_list)
a=variable(torch.tensor(1))
a
