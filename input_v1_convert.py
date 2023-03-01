import pandas as pd
import sys
import numpy as np
from tqdm import tqdm
import nltk
from nltk import tokenize
import re
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
import string

stop_words = stopwords.words('english')

def strip(s):
    sents = tokenize.sent_tokenize(s)
    for i in range(len(sents)):
        tokens = nltk.word_tokenize(sents[i])
        tokens = [re.sub(r"\d+", "", w.lower()).translate(str.maketrans('', '', string.punctuation)) for w in tokens if not w in stop_words]
        sents[i] = re.sub(' +', ' ', ' '.join(tokens).strip())
    sents = [_.lower() for _ in sents if len(_)>1]
    return '. '.join(sents)

def lower(s):
    return s.lower()


input_file = 'Data/trainData/ILDC_single_train_dev.csv'
test_file = 'Data/trainData/ILDC_single_test.csv'


# Load training data
df = pd.read_csv(input_file)


# preprocessing
df['text'] = df['text'].apply(lower)


# Indentify duplicate sentences
pos_sents = set()
for text in df.query(" label==1 ")['text']:
    pos_sents.update(tokenize.sent_tokenize(text))        
neg_sents = set()
for text in df.query(" label==0 ")['text']:
    neg_sents.update(tokenize.sent_tokenize(text))        
duplicate_sent = pos_sents.intersection(neg_sents)


# Remove duplicate sentences from input data
for i in tqdm(range(len(df))):
    text = df.iloc[i].text
    #print(text)
    sents = tokenize.sent_tokenize(text)
    #print(sents)
    new_sents = [_ for _ in sents if not _ in duplicate_sent]  
    df.at[i, 'text'] = ' '.join(new_sents)    
df.to_csv(input_file.replace('.csv', '_v1.csv'),index=False)


# Save duplicate sentences to file
duplicate_sent = list(duplicate_sent)
duplicate_sent.sort()
with open('Data/trainData/noninfo_sents.txt','w') as fp:
    for line in duplicate_sent:
        fp.write('%s\n'%line)


# Load test file
df_test = pd.read_csv(test_file)


# preprocessing
df_test['text'] = df_test['text'].apply(lower)


# Remove duplicate sentences from test data
take_out = 0
for i in tqdm(range(len(df_test))):
    text = df_test.iloc[i].text
    sents = tokenize.sent_tokenize(text)
    new_sents = [_ for _ in sents if not _ in duplicate_sent]    
    take_out += len(sents) - len(new_sents)
    df_test.at[i, 'text'] = ' '.join(new_sents)
df_test.to_csv(test_file.replace('.csv','_v1.csv'),index=False)
