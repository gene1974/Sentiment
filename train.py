import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import pickle
import time
import torch
import torch.optim as optim

from pytorchtools import EarlyStopping
from utils import logerr, logout, plot_result, label_sentence_entity
from model import NERModel
from data import *


def train_epoch(model, optimizer, train_set, dev_set = None):
    train_losses = []
    valid_losses = []
    model.train()
    for i in range(len(train_set)):
        tokens, masks, labels = train_set.get_batch(i)
        optimizer.zero_grad()
        loss = model(tokens, masks, labels) # (n_batch, n_token, n_class)
        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()

    if dev_set is None:
        return np.average(train_losses), -1
    
    model.eval()
    with torch.no_grad():
        for i in range(len(dev_set)):
            tokens, masks, labels = dev_set.get_batch(i)
            loss = model(tokens, masks, labels)
            valid_losses.append(loss.item())
    return np.average(train_losses), np.average(valid_losses)

def evaluate(model, vocabs, dev_set):
    word_list = vocabs['word_list']
    tokenizer = vocabs['tokenizer']
    tag_list = vocabs['tag_list']
    model.eval()
    correct, total = 0, 0 
    correct_num, gold_num, predict_num = 0, 0, 0
    with torch.no_grad():
        for i in range(len(dev_set)):
            tokens, masks, labels = dev_set.get_batch(i)
            # sen_len = torch.max(torch.sum(masks, dim = 1, dtype = torch.int64)).item()
            # tokens = tokens[:, :sen_len]
            # masks = masks[:, :sen_len]
            # labels = labels[:, :sen_len]
            predict = model.predict(tokens, masks)
            correct += torch.sum(predict[masks == 1] == labels[masks == 1]).item()
            total += torch.sum(masks).item()
            for j in range(labels.shape[0]):
                text = tokenizer.convert_ids_to_tokens(tokens[j], skip_special_tokens = True)
                gold_entity = label_sentence_entity(text, labels[j].tolist(), tag_list)
                pred_entity = label_sentence_entity(text, predict[j], tag_list)
                gold_num += len(gold_entity)
                predict_num += len(pred_entity)
                for entity in gold_entity:
                    if entity in pred_entity:
                        correct_num += 1
                # print(gold_entity)
                # print(pred_entity)
                # gold_entity()
        precision = correct_num / (predict_num + 1e-7)
        recall = correct_num / (gold_num + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
    logout('[Test] Acc: {:.4f} Precision: {:.4f} Recall: {:.4f} F1: {:.4f}'.format(correct / total, precision, recall, f1))
    return correct / total, precision, recall, f1
    
def savemodel(model, vocabs, model_time):
    model_path = './results/{}'.format(model_time)
    while os.path.exists(model_path):
        model_path = model_path + '_1'
    os.mkdir(model_path)
    torch.save(model.state_dict(), model_path + '/model_' + model_time)
    with open(model_path + '/vocab_' + model_time, 'wb') as f:
        pickle.dump(vocabs, f)
    logout('Save result {}'.format(model_time))

def train(args, vocabs, train_set, dev_set):
    model_time = time.strftime('%m%d%H%M', time.localtime())
    model = NERModel(args, vocabs).to('cuda')
    if args['ner_model'] == 'lstm':
        optimizer = optim.Adam(model.parameters(), lr = 1e-4)
    else:
        optimizer = optim.Adam(model.parameters(), lr = 1e-5)
    early_stopping = EarlyStopping(patience = 10, verbose = False)
    for epoch in range(args['epochs']):
        train_loss, dev_loss = train_epoch(model, optimizer, train_set, dev_set)
        if args['eval']:
            train_acc, p, r, f1 = evaluate(model, vocabs, train_set)
            dev_acc, p, r, f1 = evaluate(model, vocabs, dev_set)
            logout('[epoch {:d}] Loss: Train: {:.3f} Dev: {:.3f}, Acc: Train: {:.4f} Dev: {:.4f}'.format(epoch + 1, train_loss, dev_loss, train_acc, dev_acc))
        else:
            logout('[epoch {:d}] Loss: Train: {:.3f} Dev: {:.3f}'.format(epoch + 1, train_loss, dev_loss))
        early_stopping(dev_loss, model)
        if early_stopping.early_stop:
            logout("Early stopping")
            break
    train_acc = evaluate(model, vocabs, train_set)
    dev_acc = evaluate(model, vocabs, dev_set)
    savemodel(model, vocabs, model_time)
    return train_acc, dev_acc




if __name__ == '__main__':
    from main import parse_arg
    args = parse_arg()
    print(args)
    train_set, dev_set, test_set, vocabs = load_ner_dataset(args)
    train_loss, dev_loss = train(args, vocabs, train_set, dev_set)
    
    # test train and eval
    # model = NERModel(args, vocabs).to('cuda')
    # optimizer = optim.Adam(model.parameters(), lr = 1e-5)
    # train_loss, dev_loss = train_epoch(model, optimizer, train_set, dev_set)
    # print(train_loss, dev_loss)
    # evaluate(model, train_set, vocabs)
    # evaluate(model, dev_set, vocabs)



