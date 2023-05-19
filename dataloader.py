import os  # when loading file paths
import pandas as pd  # for lookup in annotation file
import spacy  # for tokenizer
import torch
from torch.nn.utils.rnn import pad_sequence  # pad batch
from torch.utils.data import DataLoader, Dataset
from PIL import Image  # Load img
import torchvision.transforms as transforms
import numpy as np 
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg





spacy_eng =  spacy.load("en_core_web_sm")


#inserrisco manualmete il come carattere 4 da imparrae a fine di ogni domanda
class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1

                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        list_numerized = []
        for token in tokenized_text:
            try: 
                list_numerized.append(self.stoi[token])
            except:
                list_numerized.append(self.stoi["<UNK>"])

        return list_numerized


class CocoDataset(Dataset):
    

    def __init__(self,  csv, immage_dir, freq_threshold, transform=None):
        
        #Path csv 
        self.csv = csv        # csv con immagini domande ecc        
        self.immage_dir = immage_dir         # cartella immagini coco
      
    

        #leggo i csv
        self.file_csv =pd.read_csv(csv)         # ID IMMAGINI
        

        #salvo id e domande
        
        #cvs header 
        # image_id, id, bbox, iscrowd, category_id_x, split, sentences, file_name, category_id_y, ann_id, sent_ids, ref_id

        self.id_imm_list = self.file_csv["image_id"]
        self.id = self.file_csv["id"]

        self.bbox = self.file_csv["bbox"]
        self.iscrowd = self.file_csv["iscrowd"]
        self.category_id_x = self.file_csv["category_id_x"]


        self.sentence = self.file_csv["sentences"]
        self.coco_file_name = self.file_csv["file_name"]

        self.category_id_y = self.file_csv["category_id_y"]
        self.ann_id = self.file_csv["ann_id"]
        self.sent_ids = self.file_csv["sent_ids"]
        self.ref_id = self.file_csv["ref_id"]
       

        self.transform = transform


        # Initialize vocabulary and build vocab
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.sentence.tolist())

    def __len__(self):
        return len(self.sentence)

    def __getitem__(self, index):

        frase = self.sentence[index] 
        id_imm      = self.id_imm_list[index]  
        coco_id_imm = self.coco_file_name[index] 

        correct_reference_coco = coco_id_imm.split("_")[0] + "_" + coco_id_imm.split("_")[1] + "_" + coco_id_imm.split("_")[2] + ".jpg"

        img = Image.open(os.path.join(self.immage_dir, correct_reference_coco)).convert('RGB')

        if self.transform is not None:
            img = self.transform(img) 

        numericalized_question = [self.vocab.stoi["<SOS>"]]
        numericalized_question += self.vocab.numericalize(frase)
        numericalized_question.append(self.vocab.stoi["<EOS>"])

        category = self.category_id_y[index]
        Bounding_box = self.bbox[index]

        return img, torch.tensor(numericalized_question),  len(numericalized_question), category, Bounding_box, index



      

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)

        lengths = [item[2] for item in batch]
        sorted_index = torch.argsort(torch.tensor(lengths,dtype=torch.int), descending=True)
        lengths = np.array(lengths)[sorted_index]
        
        ###
        category = [item[3] for item in batch]
        sorted_index = torch.argsort(torch.tensor(category,dtype=torch.int), descending=True)
        category = np.array(category)[sorted_index]

        Bounding_box = [item[4] for item in batch]
        
        ####


        index = [item[5] for item in batch]
        sorted_index = torch.argsort(torch.tensor(index,dtype=torch.int), descending=True)
        index = np.array(index)[sorted_index]

        imgs = imgs[sorted_index]
        targets = targets[sorted_index]

        return imgs, targets, lengths, category ,Bounding_box, index


def get_loader(
    csv,
    imm_dir , 
    freq_threshold , 
    transform,
    batch_size=32,
    num_workers=1,
    shuffle=True,
    drop_last=True,
):
    dataset  = CocoDataset(  csv, imm_dir, freq_threshold, transform)


    pad_idx = dataset.vocab.stoi["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,

        collate_fn=MyCollate(pad_idx=pad_idx),
    )

    return loader, dataset


"""
csv = r'E:\Magistrale algoritmi\Trasn-conv\cvs_file\train_data.csv'
imm_dir =r'E:\REFcoco dataset\refcocog\images'


freq_threshold = 1
############################################################################################################################################

# Declare transformations (later)
transform = transforms.Compose(
            [
                transforms.Resize((512, 512), antialias=True),
                # transforms.RandomCrop((224, 224)),
                transforms.ToTensor(),
            ]
        )

loader, dataset = get_loader(
        csv, imm_dir, freq_threshold, transform=transform , num_workers=2,
            )   


immagine, frase, len_, categoria, bbox, i = dataset.__getitem__(1)

print("stampo elemento n 2")
print(immagine )
print(frase)
print(len_)
print(categoria)
print(bbox )
print(i)
"""