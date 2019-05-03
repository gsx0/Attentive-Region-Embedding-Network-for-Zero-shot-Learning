import numpy as np
import torch

def image_load(class_file, label_file):
    with open(class_file, 'r') as f:
        class_names = [l.strip() for l in f.readlines()]
    class_map = {}
    for i,l in enumerate(class_names):
        items = l.split()
        class_map[items[-1]] = i
    #print(class_map)
    examples = []
    labels = {}
    with open(label_file, 'r') as f:
        image_label = [l.strip() for l in f.readlines()]
    for lines in image_label:
        items = lines.split()
        examples.append(items[0])
        labels[items[0]] = int(items[1])
    return examples,labels, class_map

def split_byclass(config, examples,labels, attributes, class_map):
    with open(config['train_classes'], 'r') as f:
        train_lines = [l.strip() for l in f.readlines()]
    with open(config['test_classes'], 'r') as f:
        test_lines = [l.strip() for l in f.readlines()]
    train_attr = []
    test_attr = []
    train_class_set = {}
    for i,name in enumerate(train_lines):
        idx = class_map[name]
        train_class_set[idx] = i
        # idx is its real label
        train_attr.append(attributes[idx])
    test_class_set = {}
    for i,name in enumerate(test_lines):
        idx = class_map[name]
        test_class_set[idx] = i
        test_attr.append(attributes[idx])
    train = []
    test = []
    label_map = {}
    for ins in examples:
        v = labels[ins]
        # inital label
        if v in train_class_set:
            train.append(ins)
            label_map[ins] = train_class_set[v]
        else:
            test.append(ins)
            label_map[ins] = test_class_set[v]
    train_attr = torch.from_numpy(np.array(train_attr,dtype='float')).float()
    test_attr = torch.from_numpy(np.array(test_attr,dtype='float')).float()
    return [(train,test,label_map,train_attr,test_attr)]















