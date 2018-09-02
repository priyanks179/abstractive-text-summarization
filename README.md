# abstractive-text-summarization


def genrate(inp,decoder_inp):

    enc_output,state=enc(inp)#encoder_output=[8,400,512] #state=([2,8,512])*2
    dec_inp=torch.ones(enc_output.size(0),1,dtype=torch.long).to(device)*2#2 for <sos>
    coverage = Variable(torch.zeros(dec.batch_size,dec.max_enc)).to(device)#8,400
    dec_state=dec.hid_init(state)
    cov_loss,loss=0,0
    preds=Variable(torch.zeros(args['batch'],decoder_inp.size(1)),requires_grad=True)
    for i in range(decoder_inp.size(1)):
        dec_state,p_final,coverage,attn=dec(enc_output,enc,dec_inp,inp,dec_state,coverage)
        dec_inp=torch.transpose(decoder_inp[:,i].unsqueeze(0),1,0)
        for j in range(len(torch.argmax(p_final,1))):
        #   using argmax to create prediction sentence
            preds[j,i]=torch.argmax(p_final,1)[j] 
    return(preds,p_final,attn)
  
  def train_genrator(enc_optimizer,dec_optimizer, preds):
  
    N = fake_data.size(0)
    # Reset gradients
    enc_optimizer.zero_grad()
    dec_optimizer.zero_grad()
    #   discrim is cnn classifier that predict if sentence predicted is close to real or not
    prediction = discrim(enc.embedding(pred))
    # Calculate error and backpropagate
    error = BCE_loss(prediction, ones_target(N))
    loss_genrated.backward()
    # Update weights with gradients
    enc_optimizer.step()
    dec_optimizer.step()
    # Return error
    return loss_genrated
    
    
    
class Cnn_discriminator(nn.Module):

    def __init__(self, vocab_size, embed_size,window_sizes=(3, 4, 5)):
    
        super(CnnTextClassifier, self).__init__()

        self.convs = nn.ModuleList([
            nn.Conv2d(1, 6, [window_size, embed_size], padding=(window_size - 1, 0))
            for window_size in window_sizes
        ])

        self.fc = nn.Linear(6 * len(window_sizes), 1)#2 is num of classes
        #6 is no of filters
 
    def forward(self, x):#x is 2,10,64

        # Apply a convolution + max pool layer for each window size
        x = torch.unsqueeze(x, 1)#2, 1, 10, 64 added 1 for channels
        xs = []
        for conv in self.convs:
            x2 = F.relu(conv(x))#2, 6, 14, 1
            x2 = torch.squeeze(x2, -1)#2, 6, 14
            x2 = F.max_pool1d(x2, x2.size(2))#2,6,1
            xs.append(x2)
        x = torch.cat(xs, 2) # [2,6,3] 3 is window size

        # FC
        x = x.view(x.size(0), -1)       # [2, 6 * 3]
        logits = self.fc(x)             # [2, 1] bcoz class is 1

        # Prediction
        pred=torch.sigmoid(logits)

        return pred      
