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
import validate
from rouge import Rouge 

device='cuda'

rouge = Rouge()
dir='data/batches/merge_8'
file_list = os.listdir(path=dir)



args={'hidden_size':200,'vocab_size':50000,'embed_size':64,
      'max_enc':400,'max_oovs':400,'batch':4,'max_summ':100}         

enc=model.encoder(args).to(device)
dec=model.attn_decoder(args,enc.vocab_size).to(device)


enc_optimizer=optim.Adam(enc.parameters(),lr=0.001 ,betas=(0.9, 0.999), eps=1e-08)
dec_optimizer=optim.Adam(dec.parameters(),lr=0.001 ,betas=(0.9, 0.999), eps=1e-08)
#enc_optimizer=optim.Adagrad(enc.parameters(),lr=0.1 ,initial_accumulator_value=0.1)
#dec_optimizer=optim.Adagrad(dec.parameters(),lr=0.1 ,initial_accumulator_value=0.1)
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
        
def rouge_score(decoder_inp,preds):
    inp_col=decoder_inp.size(1)
    tar_col=preds.size(1)  
          
    inp=decoder_inp.cpu().tolist()
    tar=preds.cpu().tolist()
    hyp,ref=[],[]
    if tar_col>inp_col:
        col_size=tar_col
    else:
        col_size=inp_col
    for row in range(args['batch']):
        temp1,temp2='',''
        for col in range(col_size):
            if col>=inp_col:
                temp1+=' '+str(0)
            else:
                temp1+=' '+ str(int(inp[row][col]))
            if col>=tar_col:
                temp2+=' '+str(0)
            else:
                temp2+=' '+ str(int(tar[row][col]))
        hyp.append(temp1)
        ref.append(temp2)            

    scores = rouge.get_scores(hyp, ref,avg=True)
    score_f=[]
    score_f.append(scores['rouge-1']['f'])
    score_f.append(scores['rouge-2']['f'])
    score_f.append(scores['rouge-l']['f'])
    return(score_f)    
    

iterations,print_every=75000,100
co=[]
for iter in range(iterations):
    cov_loss,loss=0,0
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
    inp=num_to_var(input[ind*args['batch']:ind*args['batch']+args['batch'],:]).to(device)#8,400
    decoder_inp=num_to_var(target[ind*args['batch']:ind*args['batch']+args['batch'],:]).to(device)
    
    enc_optimizer.zero_grad()
    dec_optimizer.zero_grad()
    enc_output,state=enc(inp)#encoder_output=[8,400,512] #state=([2,8,512])*2
    dec_inp=torch.ones(enc_output.size(0),1,dtype=torch.long).to(device)*2#2 for <sos>
    coverage = variable(torch.zeros(dec.batch_size,dec.max_enc)).to(device)#8,400
    dec_state=dec.hid_init(state)

    preds=torch.zeros(args['batch'],args['max_summ'])
    for i in range(decoder_inp.size(1)):
        dec_state,p_final,coverage,attn,p_genrate=dec(enc_output,enc,dec_inp,inp,dec_state,coverage)
        dec_inp=torch.transpose(decoder_inp[:,i].unsqueeze(0),1,0)
        cov_loss+=torch.sum(torch.min(coverage,attn)) 
        loss+=criterion(F.log_softmax(p_final,1),decoder_inp[:,i])
#        loss+=criterion(torch.log(p_final),decoder_inp[:,i]) 
       
        pred=torch.argmax(p_final,1)
        for j in range(len(pred)):
            preds[j,i]=pred[j]
        if i==0:
            p_gen=p_genrate
    dec_inp.detach()
    p_final_softmax=F.softmax(p_final,1)
#    p_final_softmax=p_final
    if torch.isnan(loss)==0 and torch.min(p_final)!=0:
        loss.backward() 
#        cov_loss.backward()
        nn.utils.clip_grad_norm_(enc.parameters(),2)
        nn.utils.clip_grad_norm_(dec.parameters(),2)
        enc_optimizer.step()
        dec_optimizer.step()
    else:
        print('loss got nan didn\'t optimized')
        print(loss)
        print('------------------------------')
        print(p_final==0)
        print('------------------------------')
#    with open('cov.txt','a') as output:
#        output.write(str(cov_loss.cpu().data.numpy().tolist()))
#        output.write('\n') 
#        co.append(cov_loss.cpu().data.numpy().tolist())
#    loss,cov_loss=loss.detach(),cov_loss.detach()
###############################################################################
    if iter%10==0:
        with open('loss.txt', 'a') as output:
            output.write(str(loss.cpu().data.numpy().tolist()))
            output.write('\n')   
        with open('param.txt','a') as handle :
            handle.write('loss '+str(loss)+'\n')  
            handle.write('coverage '+str(coverage)+'\n')  
            handle.write('coverage loss '+str(cov_loss)+'\n')  
            handle.write('pfinal min '+str(torch.min(p_final_softmax,1))+'\n')  
            handle.write('pfinal max '+str(torch.max(p_final_softmax,1))+'\n')  
            handle.write('pfinal '+str(p_final)+'\n') 
            handle.write('\n\n')
        with open('rouge.txt','a') as handle:
            score_f=rouge_score(decoder_inp,preds)
            handle.write(str(score_f[0])+' '+str(score_f[1])+' '+str(score_f[2])+'\n')
###############################################################################            
    if iter%print_every==0  :
        e=[]
        for i in dec.parameters():
            e.append(torch.max(i.grad))
        score_f=rouge_score(decoder_inp,preds)
        print('file: {} index: {} iteration: {} loss: {} max dec grad: {} '.format(count-1,ind,iter,loss,max(e)))
        print('rouge_1 f:{} rouge_2 f:{} rouge_l f:{}'.format(score_f[0],score_f[1],score_f[2]))
#        print('oov in target per sent: {} oov in pred per sent: {} p_gen: {} 1st_word: {}'.format(((decoder_inp>args['vocab_size']).to(torch.float)).sum()/args['batch'],
#             ((preds>args['vocab_size']).to(torch.float)).sum()/args['batch'],p_gen.cpu().tolist()[-1][0],preds[-1,0].to(torch.int)))
        print('------------------------------')
        summ=util.list_to_summ(preds[-1,:].cpu().numpy().tolist(),idx2oov)
        print('predicted summ : ',summ,'\n------------------------------')           
        real_summ=util.list_to_summ(decoder_inp[-1,:].cpu().numpy().tolist(),idx2oov)
        print('real summ : ',real_summ,'\n--------------------------')
        real_summ,summ,val_loss,val_cov_loss,score_f=validate.val(enc,dec,criterion,args)
        print('val loss: ',val_loss.tolist(),'\n--------------------------')
        print('rouge_1 f:{} rouge_2 f:{} rouge_l f:{}'.format(score_f[0],score_f[1],score_f[2]),'\n--------------------------')
        print('val predicted summ: ',summ,'\n------------------------------') 
        print('val real summ: ',real_summ,'\n##############################')
        with open('val.txt','a') as handle :
            handle.write(str(val_loss.tolist())+'\n')
        loss,cov_loss=0,0
        enc=enc.to(device)
        dec=dec.to(device)
        with open('val_rouge.txt','a') as handle:
            handle.write(str(score_f[0])+' '+str(score_f[1])+' '+str(score_f[2])+'\n')
    if iter%1000 ==0 and iter>0:
         torch.save(enc.state_dict(),'encoder')
         torch.save(dec.state_dict(),'decoder')  
                
    del inp,decoder_inp
    ind+=1
    
import matplotlib.pyplot as plt
plt.figure()
plt.plot(lost_list)    

#522760
#26000


s=0
for i in lost_list:
    if i>=250 and i<300 :
#    if i>300:
        s+=1

s/len(lost_list)
   
#>300 .14        .20
#200 to 250 .39  .34
#250 to 300 .30  .34
#<200 .15        .10

bins=[0,50,100,150,200,250,300,350]
plt.hist(lost_list,bins,histtype='bar',rwidth=.9)
plt.show()

for i in dec.parameters(): 
    print(i)

#for ptr nw 3    
#>300 .08
#<200 .512
# 200 to 300 .40
    
val=[]
with open('val.txt','r') as output:
    val=output.read().strip().split('\n')  
    for i,data in enumerate(val):
        val[i]=float(data)   

len(val)/4        
bins=[i*1000 for i in range(20)]
plt.hist(val,bins,histtype='bar',rwidth=.9)
plt.show() 
    

log_val=[]
for i in lost_list:
    log_val.append(np.log(i))

plt.figure()
plt.plot(log_val)

