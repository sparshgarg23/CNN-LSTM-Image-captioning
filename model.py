import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    
    
class DecoderRNN(nn.Module):
    
    def __init__(self, embed_size, hidden_size, vocab_size, drop_prob=0.2):
        super().__init__()
        self.embed_size = embed_size
        #No of units in hidden layer
        self.hidden_size = hidden_size
        #Size of vocabulary used for one-hot vectorization
        self.vocab_size = vocab_size
        # embedding layers
        self.embed = nn.Embedding(vocab_size, embed_size)
        # LSTM layer with droputout
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=2, dropout=drop_prob, batch_first=True)
        #Add an additional dropout
        self.dropout=nn.Dropout(drop_prob)
        # Linear layer that maps the hidden state output dim to the # of words as output, vocab_size
        self.dense = nn.Linear(hidden_size, vocab_size)
        
        self.softmax = nn.LogSoftmax(dim=-1)
        self.weight_set()#used to initialize weights and forget gate bias
    
    
    def forward(self, features, captions):
        
        # Initialize hidden layer
        self.batch_size=features.shape[0]
        #self.hidden=self.init_hidden(self.batch_size)
        #Ignore 'END' token and perform embedding
        captions = self.embed(captions[:,:-1])
        # Stack features and captions
        inputs = torch.cat((features.unsqueeze(1), captions), dim=1)
        #lstm_output
        lstm_out, self.hidden = self.lstm(inputs)
        #Apply dropout
        lstm_out=self.dropout(lstm_out)
        # Fully connected layer to turn the output into vectors in the size  (batch_size, caption length, vocab_size)
        output = self.dense(lstm_out)
        #compute softmax
        output_scores = self.softmax(output)
        return output_scores
    def weight_set(self):
        #Do weight initilization and add bias to forget gate
        self.dense.bias.data.fill_(0.01)
        torch.nn.init.xavier_normal_(self.dense.weight)
        for names in self.lstm._all_weights:
            for name in filter(lambda n:"bias" in n,names):
                bias=getattr(self.lstm,name)
                n=bias.size(0)
                start,end=n//4,n//4
                bias.data[start:end].fill_(1.0)
    
    def sample(self, inputs, states=None, max_len=20):
        sampled_ids = [] 
        for i in range(max_len): 
            hiddens, states = self.lstm(inputs, states) 
            outputs = self.dense(hiddens.squeeze(1)) 
            predicted = outputs.max(1)[1] 
            if predicted==1:
                break
            sampled_ids.append(predicted.data[0].item()) 
            inputs = self.embed(predicted) 
            inputs = inputs.unsqueeze(1) 
        return sampled_ids

