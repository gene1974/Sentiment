import torch
from utils import logerr, logout, plot_result, label_sentence_entity

def evaluate_ner(model, dev_set, vocabs):
    word_list = vocabs['word_list']
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
                text = [word_list[int(w)] for w in tokens[j][masks[j] == 1]]
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
        precision = correct_num / (predict_num + 0.000000001)
        recall = correct_num / (gold_num + 0.000000001)
        f1 = 2 * precision * recall / (precision + recall + 0.000000001)
        logout('[Test] Tagging accuracy: {:.4f}, {} tokens'.format(correct / total, total))
        logout('[Test] Precision: {:.4f} Recall: {:.4f} F1: {:.4f}'.format(precision, recall, f1))
    logout('Tagging accuracy: {}'.format(correct / total))