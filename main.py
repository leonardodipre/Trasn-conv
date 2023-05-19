import pandas as pd
import torch
from torch import nn, optim
from  model  import Encoder_SelfAtt_transBlock, FeatureExtractor, Encoder_decoder
from dataloader import get_loader
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm
from loss import loss_function
import ast

def train():


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)


    # Declare transformations (later)
    transform = transforms.Compose(
            [
                transforms.Resize((512, 512), antialias=True),
                # transforms.RandomCrop((224, 224)),
                transforms.ToTensor(),
            ]
        )

############################################################################################################################################
                                                        #DIRECTORY#
    
    csv = r'E:\Magistrale algoritmi\Trasn-conv\cvs_file\overfint_10.csv'
    imm_dir =r'E:\REFcoco dataset\refcocog\images'


    freq_threshold = 1
############################################################################################################################################


    loader, dataset = get_loader(
            csv, imm_dir, freq_threshold, transform=transform , num_workers=2,shuffle= False
            )   

    

    #Hyperparameters
    embed_size = 224
    vocab_size = len(dataset.vocab)
    learning_rate = 1e-3
    num_epochs = 20
  
    hidden_dim = 64
    heads = 8
    forward_expansion = 4
    dropout = 0

    
    #init model
    model = Encoder_decoder(embed_size, heads, hidden_dim, vocab_size, forward_expansion, dropout)

    
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for j in range(250):
        for  i, (imm, frase, lengths, category , bbox, index) in  enumerate(tqdm(loader)):
        
           
            

            imm = imm.to(device)
            frase = frase.to(device)

        
            mask =None
            out  = model(imm, frase, mask)
            
            
        
            # bbox è str trasformiamo in un tesnore di dim [32,4]
            bbox_matrix = torch.empty(1,4)
            for item in bbox:
                number_list = ast.literal_eval(item)
                flattened_list = np.array(number_list).flatten()
                tensor = torch.tensor(flattened_list).unsqueeze(0)
                bbox_matrix = torch.cat((bbox_matrix, tensor), dim=0)

            bbox_matrix = torch.index_select(bbox_matrix, dim=0, index=torch.arange(1, bbox_matrix.size(0)))
            ###############################################################################

            loss = loss_function(out, bbox_matrix, 5, 2)
        
            
            
            print("Box pred ...")
            print(" ")
            print("Ground", bbox_matrix[1])
            print("Preds", out[1])
            print(" ################")
            print("Box pred ...")
            print(" ")
            print("Ground", bbox_matrix[2])
            print("Preds", out[2])
            print(" ################")
            print("Box pred ...")
            print(" ")
            print("Ground", bbox_matrix[3])
            print("Preds", out[3])
            print(" ################")
            print(loss)

            print("")
            print("")
            print("")
            print("")
            optimizer.zero_grad()
            loss.backward()
            

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

       

            
        

            
            
                

       
       

        


        

if __name__ == "__main__":
    train()