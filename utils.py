import os
import pickle
import torch
import numpy as np
from torch.autograd import Variable, Function
from random import sample

def save(toBeSaved, filename, mode='wb'):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    file = open(filename, mode)
    pickle.dump(toBeSaved, file)
    file.close()

def load(filename, mode='rb'):
    file = open(filename, mode)
    loaded = pickle.load(file)
    file.close()
    return loaded

def pad_sents(sents, pad_token):
    sents_padded = []
    lens = get_lens(sents)
    max_len = max(lens)
    sents_padded = [sents[i] + [pad_token] * (max_len - l) for i, l in enumerate(lens)]
    return sents_padded

def sort_sents(sents, reverse=True):
    sents.sort(key=(lambda s: len(s)), reverse=reverse)
    return sents

def get_mask(sents, unmask_idx=1, mask_idx=0):
    lens = get_lens(sents)
    max_len = max(lens)
    mask = [([unmask_idx] * l + [mask_idx] * (max_len - l)) for l in lens]
    return mask

def get_lens(sents):
    return [len(sent) for sent in sents]

def get_max_len(sents):
    max_len = max([len(sent) for sent in sents])
    return max_len

def truncate_sents(sents, length):
    sents = [sent[:length] for sent in sents]
    return sents

def get_loss_weight(labels, label_order):
    nums = [np.sum(labels == lo) for lo in label_order]
    loss_weight = torch.tensor([n / len(labels) for n in nums])
    return loss_weight

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def to_np(x):
    return x.data.cpu().numpy()

def select(train, selec_indices):
    temp = []
    for i in range(len(train)):
        print("length is "+str(len(train[i])))
        print(i)
        #print(train[i])
        ele = list(train[i])
        temp.append([ele[i] for i in selec_indices])
    return temp

def make_weights_for_balanced_classes(event, nclasses = 15):
    count = [0] * nclasses
    for item in event:
        count[item] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(event)
    for idx, val in enumerate(event):
        weight[idx] = weight_per_class[val]
    return weight

def split_train_validation(train, percent):
    whole_len = len(train[0])

    train_indices = (sample(range(whole_len), int(whole_len * percent)))
    train_data = select(train, train_indices)
    print("train data size is "+ str(len(train[3])))
    # print()

    validation = select(train, np.delete(range(len(train[0])), train_indices))
    print("validation size is "+ str(len(validation[3])))
    print("train and validation data set has been splited")

    return train_data, validation
