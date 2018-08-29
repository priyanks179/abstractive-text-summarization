import os
import random

dir1='data/batches/cnn_8'
dir2='data/batches/dm_8'
output_dir='data/batches/merge_8'

paths=[]
file_list = os.listdir(path=dir1)
for file in file_list:
    path=os.path.join(dir1,file)
    paths.append(path)
file_list = os.listdir(path=dir2)
for file in file_list:
    path=os.path.join(dir2,file)
    paths.append(path)

random.shuffle(paths)      

for index,path in enumerate(paths):
    with open(path,'r',encoding='utf8') as f:
        txt=f.read()
        batch='batch_'+str(index)+'.txt'
        path=os.path.join(output_dir,batch)
        with open(path,'w',encoding='utf8') as g:
            g.write(txt)
    print(batch+' completed')            
 
