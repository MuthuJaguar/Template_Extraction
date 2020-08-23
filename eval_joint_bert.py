
from readers.goo_format_reader import Reader
from vectorizers.bert_vectorizer import BERTVectorizer
from models.joint_bert import JointBertModel
from utils import flatten
from prettytable import PrettyTable

import os
import pickle
import tensorflow as tf
import numpy as np
from sklearn import metrics
from pathlib import Path


VALID_TYPES = ['bert', 'albert']

load_folder_path = Path('F:/Deep_Learning/SID/saved_models/joint_bert_model')
data_folder_path = Path('F:/Deep_Learning/SID/data/snips/test')
batch_size = 128
type_ = 'bert'


if type_ == 'bert':
    bert_model_hub_path = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"
    is_bert = True
elif type_ == 'albert':
    bert_model_hub_path = "https://tfhub.dev/tensorflow/albert_en_base/1"
    is_bert = False
else:
    raise ValueError('type must be one of these values: %s' % str(VALID_TYPES))
    
bert_vectorizer = BERTVectorizer(is_bert, bert_model_hub_path)

# loading models
print('Loading models ...')
if not os.path.exists(load_folder_path):
    print('Folder `%s` not exist' % load_folder_path)

with open(os.path.join(load_folder_path, 'tags_vectorizer.pkl'), 'rb') as handle:
    tags_vectorizer = pickle.load(handle)
    slots_num = len(tags_vectorizer.label_encoder.classes_)
    slots = tags_vectorizer.label_encoder.classes_
with open(os.path.join(load_folder_path, 'intents_label_encoder.pkl'), 'rb') as handle:
    intents_label_encoder = pickle.load(handle)
    intents_num = len(intents_label_encoder.classes_)
    intents = intents_label_encoder.classes_
    
model = JointBertModel.load(load_folder_path)


data_text_arr, data_tags_arr, data_intents = Reader.read(data_folder_path)
data_input_ids, data_input_mask, data_segment_ids, data_valid_positions, data_sequence_lengths = bert_vectorizer.transform(data_text_arr)

def get_results(input_ids, input_mask, segment_ids, valid_positions, sequence_lengths, tags_arr, 
                intents, tags_vectorizer, intents_label_encoder):
    predicted_tags, predicted_intents = model.predict_slots_intent(
            [input_ids, input_mask, segment_ids, valid_positions], 
            tags_vectorizer, intents_label_encoder, remove_start_end=True)
    gold_tags = [x.split() for x in tags_arr]
    data_tags = [x.split() for x in data_text_arr]
    #print(metrics.classification_report(flatten(gold_tags), flatten(predicted_tags), digits=3))
    f1_score = metrics.f1_score(flatten(gold_tags), flatten(predicted_tags), average='micro')
    acc = metrics.accuracy_score(intents, predicted_intents)
    return f1_score, acc, predicted_tags, predicted_intents, gold_tags, data_tags

print('==== Evaluation ====')
f1_score, acc, tag_pred, intent_pred, gold_tags, data_tags = get_results(data_input_ids, data_input_mask, data_segment_ids, data_valid_positions,
                            data_sequence_lengths, 
                            data_tags_arr, data_intents, tags_vectorizer, intents_label_encoder)
print('Slot f1_score = %f' % f1_score)
print('Intent accuracy = %f' % acc)

tf.compat.v1.reset_default_graph()

data_tags = [x.split() for x in data_text_arr]

res = [0]*len(data_text_arr)
for i in range(0,len(data_text_arr)):
    res[i] = list(zip(data_tags[i], tag_pred[i]))


y = []
x = PrettyTable()

x.field_names = slots

for i in range(0, len(tag_pred)):
    z = []
    l = tag_pred[i]
    for j in slots:
        k = j
        if j in l:
            z.append(j)
        else:
            z.append(0)
    y.append(z)

result = []           
for i in range(0,len(res)):
    first = res[i]
    second = y[i]
    third = y[i]
    fourth = np.array(first)
    for i in range(0,len(first)):
        for j in range(0,len(second)):
            if fourth[i][1] == second[j]:
                third[j] = fourth[i][0]
    result.append(third)


for i in range(len(res)):
    x.add_row(result[i])
    
table_txt = x.get_string()
with open('output.txt','w') as file:
    file.write(table_txt)






