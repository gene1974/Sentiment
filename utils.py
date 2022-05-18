
import logging
import matplotlib.pyplot as plt
import sys
import time

def logerr(content):
    logging.getLogger('matplotlib.font_manager').disabled = True
    log_format = '[%(asctime)s] %(message)s'
    date_format = '%Y-%m%d %H:%M:%S'
    logging.basicConfig(level = logging.DEBUG, format = log_format, datefmt = date_format)
    logging.info(content)

def logout(content):
    logging.getLogger('matplotlib.font_manager').disabled = True
    log_format = '[%(asctime)s] %(message)s'
    date_format = '%Y-%m%d %H:%M:%S'
    logging.basicConfig(stream = sys.stdout, level = logging.DEBUG, format = log_format, datefmt = date_format)
    logging.info(content)

def plot_result(avg_train_losses, avg_valid_losses):
    model_time = '{}'.format(time.strftime('%m%d%H%M', time.localtime()))
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(avg_train_losses)
    plt.plot(avg_valid_losses)
    plt.legend(['train_loss', 'valid_loss'])
    plt.savefig('./train_loss/' + model_time + '.png')

def label_sentence_entity(text, tags, tag_list):
    tags = [tag_list[i] for i in tags]
    entity = []
    count = len(text)
    i = 0
    while i < count:
        if tags[i][0] == 'B':
            j = i + 1
            while j < count:
                if tags[j][0] == 'E':
                    break
                else:
                    j += 1
            entity.append({
                "text": ''.join(text[i: j]),
                "start_index": i,
                "end_index": j,
                "label": tags[i][2:]
            })
            i = j + 1
        elif tags[i][0] == 'S':
            entity.append({
                "text": text[i],
                "start_index": i,
                "end_index": i,
                "label": tags[i][2:]
            })
            i += 1
        else:
            i += 1
    return entity

# print(tokens[0], labels[0])
# label_sentence_entity()





