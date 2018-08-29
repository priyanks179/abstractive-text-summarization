# abstractive-text-summarization


       #----------------------------------------------------------------------
       ##here inp is input array in encoder and attn is attention over encoder's input
       ## objective of this code is to create  sparse tensor p_copy of[batch,dec_vocab_size+oov_size]  (out of vocablury)
       ##having  encoder input indexes marked with their respective  attention in p_copy
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
        
        
        #---------------------------------------------------------------
  ## this is part of code in model's attn_decoder which is taking lot of time to compute 
  
  ##link to dataset 
  https://drive.google.com/drive/folders/113Nv7tXAtUJmOELR_2O7bY_NcD64cYSt
  
