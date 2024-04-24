import os, re
import csv
import nltk
import pickle
from collections import Counter
import numpy as np
from tqdm import tqdm
import torch.utils.data
from torchvision.datasets import ImageFolder
from PIL import Image
from torchvision import transforms
import json
import pickle as pkl
import pandas as pd
from transformers import PreTrainedTokenizerFast
from typing import Union
import numpy as np
from numpy.random import Generator, PCG64
import re
import random

import os, pickle
from tqdm import tqdm
import numpy as np
import torch.utils.data
import PIL
from random import randrange
import json
from torchvision import transforms
from PIL import Image
a = ['cave', 'tree', 'lake', 'pool', 'zoo', 'desert', 'ocean', 'mountain', 'river', 'forest', 'hill', 'stage', 
     'bowling alley', 'restaurant', 'golf course', 'gym', 'jail', 'circus', 'hospital', 'television', 'police station', 'theater',
     'car', 'boat', 'airplane', 'bus', 'train', 'plain', 'ship', 'police car', 'taxi', 'outside', 'yard', 'road', 'quarry', 
     'park', 'backyard', 'outdoors', 'sidewalk','beach', 'front yard', 'street', 'sky', 'back yard',
     'office', 'window', 'store', 'garage', 'inside', 'front door', 'hallway', 'hotel', 'room', 'living room', 
     'doorway', 'kitchen', 'bedroom', 'dining room', 'bathroom', 'livingroom', 'home']

def get_background_concepts(setting):
    
    import re
    import numpy as np

    #concepts = [["house"], ["door"], ["table"], ["car"], ["snow"], ["sky"], ["tree", "forest", "woods"]]
    concepts = [['room', 'living room', 'doorway', 'kitchen', 'bedroom', 'dining room', 'bathroom', 'livingroom', 'home'], 
                ['office', 'window', 'store', 'garage', 'inside', 'front door', 'hallway', 'hotel'], 
                ['outside', 'yard', 'road', 'quarry', 'park', 'backyard', 'outdoors', 'sidewalk','beach', 'front yard', 'street', 'sky', 'back yard'], 
                ['car', 'boat', 'airplane', 'bus', 'train', 'plain', 'ship', 'police car', 'taxi'], 
                ['stage', 'bowling alley', 'restaurant', 'golf course', 'gym', 'jail', 'circus', 'hospital', 'television', 'police station', 'theater'], 
                ['cave', 'tree', 'lake', 'pool', 'zoo', 'desert', 'ocean', 'mountain', 'river', 'forest', 'hill']]

    #caption = re.split(pattern=",|\.| ", string=caption.strip().lower())

    res = []
    for concept in concepts:
        is_present = 0
        if setting in concept:
            is_present = 1.
        res.append(is_present)
    
    if sum(res) == 0:
        res.append(1)
    else:
        res.append(0)

    return np.array(res)


class StoryImageDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            img_folder, 
            tokenizer:Union[PreTrainedTokenizerFast,int], 
            preprocess, 
            mode='train', 
            video_len=4, 
            out_img_folder=None, 
            return_labels=False, 
            size=None, 
            return_captions=False, 
            return_tokenized_captions=False,
            character_names=["Wilma", "Fred", "Betty", "Barney", "Dino", "Pebbles", "Mr Slate"], #changed
            max_sentence_length=100,
            text_encoder=None,
            eval_classifier=False,
            seed=False,
            character_emphasis=False,
            return_token_indices=False,
            use_chatgpt_captions=False
    ):
        self.max_sentence_length = max_sentence_length
        self.lengths = []
        self.followings = []
        self.images = []
        #self.img_dataset = ImageFolder(img_folder)
        self.img_folder = img_folder
        self.labels = pickle.load(open(os.path.join(self.img_folder, 'labels.pkl'), 'rb')) # changed
        self.video_len = video_len
        
        #self.descriptions_original = np.load(os.path.join(img_folder, 'descriptions.npy'), allow_pickle=True, encoding='latin1').item()
        #self.descriptions = pkl.load(open(os.path.join(img_folder, 'descriptions_vec_512.pkl'), 'rb'))
        
        self.return_captions = return_captions
        self.return_tokenized_captions = return_tokenized_captions
        self.character_names = character_names
        self.eval_classifier = eval_classifier
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.return_labels = return_labels
        self.out_img_folder = out_img_folder
        self.return_token_indices = return_token_indices
        self.use_chatgpt_captions = use_chatgpt_captions

        if seed:
            self.rng = Generator(PCG64(seed=42))
        else:
            self.rng = Generator(PCG64())

        if os.path.exists(os.path.join(img_folder, 'img_cache4.npy')) and os.path.exists(os.path.join(img_folder, 'following_cache4.npy')):
            # image pathes = 1
            self.images = np.load(os.path.join(img_folder, 'img_cache4.npy'), encoding='latin1')
            # image pathes of next four = 2, 3, 4, 5
            self.followings = np.load(os.path.join(img_folder, 'following_cache4.npy'))
            self.counter = ''

        if os.path.exists(os.path.join(self.img_folder, 'following_cache' + str(video_len) +  '.pkl')):
            self.followings = pickle.load(open(os.path.join(self.img_folder, 'following_cache' + str(video_len) + '.pkl'), 'rb'))
        

        splits = json.load(open(os.path.join(self.img_folder, 'train-val-test_split.json'), 'r'))
        train_id, val_id, test_id = splits["train"], splits["val"], splits["test"]
#       train_id = train_id[:5]

        train_id = [tid for tid in train_id if tid in self.followings]
        val_id = [vid for vid in val_id if vid in self.followings]
        test_id = [tid for tid in test_id if tid in self.followings]

        

        if mode == 'train':
            self.ids = train_id
        elif mode =='val':
            val_id = [vid for vid in val_id if len(self.followings[vid]) == video_len]
            self.ids = val_id
        elif mode == 'test':
            test_id = [vid for vid in test_id if len(self.followings[vid]) == video_len]
            self.ids = test_id[:1900]
        else:
            raise ValueError
        #print("Total number of clips {}".format(len(self.orders)))
        
        if size:
            self.ids = [self.ids[i] for i in range(size)]
        
        self.annotations = json.load(open(os.path.join(self.img_folder, 'flintstones_annotations_v1-0.json')))
        self.descriptions = {}
        for sample in self.annotations:
            self.descriptions[sample["globalID"]] = sample["description"]

        

    def __len__(self):
        return len(self.ids) 

    def sample_image(self, im):
        shorter, longer = min(im.size[0], im.size[1]), max(im.size[0], im.size[1])
        video_len = int(longer/shorter)
        # se = np.random.randint(0,video_len, 1)[0]
        se = self.rng.integers(0,video_len, 1)[0]
        return im.crop((0, se * shorter, shorter, (se+1)*shorter))
        
    
    
    def __getitem__(self, item):

        globalIDs = [self.ids[item]] + self.followings[self.ids[item]]
        images = []
        for globalID in globalIDs:
            path = os.path.join(self.img_folder, 'video_frames_sampled', globalID + '.npy')
            
            arr = np.load(path)
            n_frames = arr.shape[0]
            random_range = randrange(n_frames)
            im = arr[random_range]
            #image = np.array(im)
            image = PIL.Image.fromarray(im.astype('uint8'), 'RGB')
            #print(type(image))
            images.append(self.preprocess(self.sample_image(image)))
        
        labels = [torch.tensor(self.labels[globalID]) for globalID in globalIDs]
            
        if self.eval_classifier:
            if self.return_captions:
                # get captions
                #image_captions = [self.descriptions_original[src_img_path_id]]+[self.descriptions_original[other_id] for other_id in tgt_img_ids]
                #image_captions = [desc[0].lower() for desc in image_captions]
                #return torch.stack(images), torch.tensor(np.vstack(labels)), masks, image_captions
                pass
            return torch.stack(images), torch.tensor(np.vstack(labels))
        
        text_embeddings = []
        for globalID in globalIDs:
            text = self.descriptions[globalID]
            #print(text)
            tokenized_text = self.tokenizer(text, padding=True, return_tensors='pt', truncation=True, max_length=self.max_sentence_length).input_ids
            #print(tokenized_text.shape)
            desired_size = (1, self.max_sentence_length)
            padded_tensor = torch.nn.functional.pad(tokenized_text, 
                                                    pad=(0, desired_size[1] - tokenized_text.size(1), 0, desired_size[0] - tokenized_text.size(0)),
                                                    value = 1)
            text_embeddings.append(padded_tensor)
        
        
        concept_labels = []
        self.settings = {}
        for sample in self.annotations:
            self.settings[sample["globalID"]] = sample["setting"]
        
        for globalID in globalIDs:
            concept_lebel = get_background_concepts(self.settings[globalID])
            concept_labels.append(torch.tensor(concept_lebel))


        masks = torch.tensor(1)
        
        if self.return_token_indices:
            pass
        else:
            return torch.stack(images), torch.squeeze(torch.stack(text_embeddings), dim=1), torch.tensor(np.vstack(labels)), torch.stack(concept_labels), masks
