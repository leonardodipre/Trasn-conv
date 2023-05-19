import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F



class CNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(CNN,self).__init__()

        self.train_CNN = train_CNN
        self.resNet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.preprocess = ResNet18_Weights.DEFAULT.transforms()

        #set the last layer as embed_size
        self.resNet.fc = nn.Linear(self.resNet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resNet(self.preprocess(images))
        return features

#Soft parser for wxtracting token form a tensor of the tokenized phrase
class SoftParser(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(SoftParser, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True)
        self.fc = nn.Linear(2 * hidden_dim, 1)
        
    def forward(self, x):
        # x is a tensor of shape (seq_len, batch_size), where seq_len is the length of the input sequence
        # and batch_size is the size of the input batch
        
        embedded = self.embedding(x) # embedded has shape (seq_len, batch_size, embed_dim)

        
        lstm_out, _ = self.lstm(embedded) # lstm_out has shape (seq_len, batch_size, 2 * hidden_dim)
        
       
        # Compute attention weights
        attention_weights = F.softmax(self.fc(lstm_out), dim=1) # attention_weights has shape (seq_len, batch_size, 1)
        
        # Compute textual tokens
        textual_tokens = torch.sum(embedded * attention_weights, dim=1) # textual_tokens has shape (batch_size, embed_dim)
        
        return textual_tokens




class SelfAttention(nn.Module):
    
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)

        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask, flag = True):
        # Get number of training examples
        N = query.shape[0]

        len_val = values.shape[1]

        values = self.values(values)  # (N, value_len, embed_size)
        keys = self.keys(keys)  # (N, key_len, embed_size)
        queries = self.queries(query)  # (N, query_len, embed_size)

       
        
        values = values.reshape(N, self.heads, self.head_dim)
        keys = keys.reshape(N, self.heads, self.head_dim)
        queries = queries.reshape(N, self.heads, self.head_dim)
        
       

        energy = torch.einsum("ijk , ijk->ijk", [queries, keys])
      

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=2)

        out = torch.einsum("nql,nhd->nhd", [attention, values]).reshape(
            N, self.heads * self.head_dim
        )

        if flag == True:
            out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, embed_size)

        return out 

           
class FeatureExtractor(nn.Module):
    def __init__(self , embde, hidden_dim, vocab_size):
        super(FeatureExtractor,self).__init__()

        self.cnn = CNN(embde, train_CNN=False)
        self.lstm = SoftParser(vocab_size, embde, hidden_dim)

    def forward(self, image, caption):

        IMM_features = self.cnn(image)
        Text_features = self.lstm(caption)

        return IMM_features, Text_features






class Encoder_SelfAtt_transBlock(nn.Module):
    def __init__(self , embed_size, heads, hidden_dim, vocab_size, forward_expansion, dropout):
        super(Encoder_SelfAtt_transBlock,self).__init__()

        self.extractor = FeatureExtractor(embed_size, hidden_dim, vocab_size)
        
        
        self.attention = SelfAttention(embed_size, heads)
        
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

        self.fc =  nn.Linear(embed_size, embed_size)

        
        
    def forward(self, image, caption , mask):

        #estraggo le features
        imm_feture , text_fetures = self.extractor(image, caption)

        #hanno size di [32 , 224] entrambi
        
        #Calcolo i valori dalle fature estratte per passare ai blocchi di self_attention
        q_imm , k_imm , v_imm = self.fc(imm_feture), self.fc(imm_feture) , self.fc(imm_feture)

        q_text , k_text , v_text = self.fc(text_fetures), self.fc(text_fetures) , self.fc(text_fetures)


        
        
        #Parte Trasformer
       
        attention_text = self.attention(v_text, k_text, q_text, mask)
        # Add skip connection, run through normalization and finally dropout

        
        x = self.dropout(self.norm1(attention_text + q_text))
        forward = self.feed_forward(x)
        out_text = self.dropout(self.norm2(forward + x))

        #parte immagini 

        #calcolo qv passando il testo
        qv = self.attention(out_text, out_text, q_imm, mask, False)
        q_imm = qv + q_imm

        attention_immage = self.attention(v_imm, k_imm, q_imm, mask)
        y = self.dropout(self.norm1(attention_immage + q_imm))       
        forward_imm = self.feed_forward(y)

        out_imm = self.dropout(self.norm2(forward_imm + y))

        return out_text, out_imm
       

class Decoder(nn.Module):
    def __init__(self , embed_size, heads, forward_expansion, dropout):
        super(Decoder,self).__init__()

        self.attention = SelfAttention(embed_size, heads)
        
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

        self.fc =  nn.Linear(embed_size, embed_size)


    def forward(self, out_E_text, out_E_immage, mask):
            
            #prima parte passo la query del testo come grounding queris al primo blocc Attention
        q_text , k_text , v_text = self.fc(out_E_text), self.fc(out_E_text) , self.fc(out_E_text)

        q_imm , k_imm , v_imm = self.fc(out_E_immage), self.fc(out_E_immage) , self.fc(out_E_immage)


            #funzione softmax orende in input qg, Kg, vg from G
        attention_text = self.attention(v_text, k_text, q_text, mask) 
            #Normalizzaimo con lo skip 
        x = self.dropout(self.norm1(attention_text + q_text))    


            #seconde parte altro selfAttention aggiugiamo le immagini 

            #preden in input q da text e K e V da visual 
        attention_immage = self.attention(v_imm, k_imm, x, mask) 

        y = self.dropout(self.norm1(attention_immage + x))       
        forward_imm = self.feed_forward(y)

        final_out = self.dropout(self.norm2(forward_imm + y))

        return final_out


class Encoder_decoder(nn.Module):
    def __init__(self , embed_size, heads, hidden_dim, vocab_size, forward_expansion, dropout):
        super(Encoder_decoder,self).__init__()

        self.encoder = Encoder_SelfAtt_transBlock(embed_size, heads, hidden_dim, vocab_size, forward_expansion, dropout)
        self.decoder = Decoder(embed_size, heads, forward_expansion, dropout)

        #prediction head
        self.fc1 = nn.Linear(embed_size, embed_size, bias=False)
        self.relu = nn.ReLU()
       
        self.fc2 = nn.Linear(embed_size, 4, bias=False)  # 4 outputs: center_x, center_y, width, height

    def forward(self, immage, text, mask):
        
        out_text, out_imm = self.encoder(immage, text, mask)
        print("Out dal encoder")
        out_final = self.decoder(out_text, out_imm, mask)


        out_final = self.fc1(out_final)
        out_final = self.fc2(out_final)
        out_final = self.relu(out_final)

        return out_final




        