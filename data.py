# -----------------------------------------------------------
# "Exploring a Fine-Grained Multiscale Method for Cross-Modal Remote Sensing Image Retrieval"
# Yuan, Zhiqiang and Zhang, Wenkai and Fu, Kun and Li, Xuan and Deng, Chubo and Wang, Hongqi and Sun, Xian
# IEEE Transactions on Geoscience and Remote Sensing 2021
# Writen by YuanZhiqiang, 2021.  Our code is depended on MTFN
# ------------------------------------------------------------

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import nltk
import numpy as np
import yaml
import argparse

from transformers import BertTokenizer

import utils
from vocab import deserialize_vocab
from PIL import Image


class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    """

    def __init__(self, data_split, vocab, opt):
        self.vocab = vocab
        self.loc = opt['dataset']['data_path']
        self.img_path = opt['dataset']['image_path']

        # Captions
        self.captions = []
        self.maxlength = 0

        if data_split != 'test':
            with open(self.loc + '%s_caps_verify.txt' % data_split, 'rb') as f:
                for line in f:
                    self.captions.append(line.strip())

            self.images = []
            with open(self.loc + '%s_filename_verify.txt' % data_split, 'rb') as f:
                for line in f:
                    self.images.append(line.strip())
        else:
            with open(self.loc + '%s_caps.txt' % data_split, 'rb') as f:
                for line in f:
                    self.captions.append(line.strip())

            self.images = []
            with open(self.loc + '%s_filename.txt' % data_split, 'rb') as f:
                for line in f:
                    self.images.append(line.strip())

        self.length = len(self.captions)
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        if len(self.images) != self.length:
            self.im_div = 5
        else:
            self.im_div = 1

        if data_split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((278, 278)),
                transforms.RandomRotation(0, 90),
                transforms.RandomCrop(256),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index // self.im_div
        caption = self.captions[index]

        image = Image.open(self.img_path + str(self.images[img_id])[2:-1]).convert('RGB')
        image = self.transform(image)  # torch.Size([3, 256, 256])

        return image, caption, index, img_id

    def __len__(self):
        return self.length


class Collate:
    def __init__(self, opt):
        self.tokenizer = BertTokenizer.from_pretrained(opt['text-model']['bert-text'])

    def __call__(self, data):
        # Sort a data list by caption length
        data.sort(key=lambda x: len(x[1]), reverse=True)
        images, captions, ids, img_ids = zip(*data)

        # Merge images (convert tuple of 3D tensor to 4D tensor)
        images = torch.stack(images, 0)

        # Merget captions (convert tuple of 1D tensor to 2D tensor)

        cap_lengths = [len(self.tokenizer.tokenize(str(c))) + 2 for c in
                       captions]  # + 2 in order to account for begin and end tokens
        max_len = max(cap_lengths)
        captions_ids = [torch.LongTensor(self.tokenizer.encode(str(c), max_length=max_len, pad_to_max_length=True))
                        for c in captions]
        captions = captions_ids
        targets = torch.zeros(len(captions), max(cap_lengths)).long()
        for i, cap in enumerate(captions):
            end = cap_lengths[i]
            targets[i, :end] = cap[:end]

        return images, targets, cap_lengths, ids


def get_precomp_loader(data_split, vocab, batch_size=100,
                       shuffle=True, num_workers=0, opt={}):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PrecompDataset(data_split, vocab, opt)
    collate_fn = Collate(opt)
    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=False,
                                              collate_fn=collate_fn,
                                              num_workers=num_workers)
    return data_loader


def get_loaders(vocab, opt):
    train_loader = get_precomp_loader('train', vocab,
                                      opt['dataset']['batch_size'], True, opt['dataset']['workers'], opt=opt)
    val_loader = get_precomp_loader('val', vocab,
                                    opt['dataset']['batch_size_val'], False, opt['dataset']['workers'], opt=opt)
    return train_loader, val_loader


def get_test_loader(vocab, opt):
    test_loader = get_precomp_loader('test', vocab,
                                     opt['dataset']['batch_size_val'], False, opt['dataset']['workers'], opt=opt)
    return test_loader
