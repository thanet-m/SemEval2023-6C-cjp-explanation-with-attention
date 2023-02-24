from os.path import join

import numpy as np
import nltk
from arenets.external.readers.pandas_csv_reader import PandasCsvReader
from tqdm import tqdm

from common import INPUT_DIR
from processing.fix_words import fix_words
from processing.pmi import calc_pmi_for_terms
from processing.sentence_splitter import split_text_on_sentences


def reg(w, d):
    if w not in d:
        d[w] = 1
    else:
        d[w] += 1


def clean_sentence(s, do_pos=True):
    s = s[:-1]
    s = s.replace(',', '').replace('- ', ' ')

    splitted_words = s.split()
    data_iter = nltk.pos_tag(splitted_words) if do_pos else splitted_words

    words = []
    for data in data_iter:

        if do_pos:
            w, pos = data

            # Pick only verbs.
            if ("VB" not in pos) and do_pos:
                continue
        else:
            w = data

        failed = False
        for c in w:
            if c.isdigit() or c == '.':
                failed = True
                continue
        if not failed:
            words.append(w)

    return ' '.join(words)


def create_texts(df, keep_labels=False):
    texts = []
    for i, r in tqdm(df.iterrows(), desc="Create texts"):
        text = r["text"]
        text = ' '.join(text.split())
        text = fix_words(text)
        s = split_text_on_sentences(text)
        texts.append(s if not keep_labels else (s, r["label"]))
    return texts


reader = PandasCsvReader()
storage = reader.read(join(INPUT_DIR, "sample-orig-train.tsv.gz"))
df_train = storage._df
print(df_train.head())

texts = create_texts(df_train, keep_labels=True)

n_grams_total = {}
total = len(df_train)
dec = 0
acc = 0
n_grams_acc = {}
n_grams_dec = {}
for sentences, label in tqdm(texts):
    dec += label == 0
    acc += label == 1
    n_grams_c = n_grams_acc if label == 1 else n_grams_dec
    for s in [clean_sentence(s) for s in sentences]:
        for w in s.split():
            reg(w, n_grams_total)
            reg(w, n_grams_c)

# calc so
total = sum(n_grams_total.values())
acc_total = sum(n_grams_acc.values())
dec_total = sum(n_grams_dec.values())
acc_prob_total = acc / (acc+dec)
dec_prob_total = dec / (acc+dec)
so = {}

# How much we get
p = 0.75
get_acc_count = round(len(n_grams_acc) * p)
get_dec_count = round(len(n_grams_dec) * p)
acc_items = list(n_grams_acc.items())
dec_items = list(n_grams_dec.items())
k_acc = sorted(acc_items, key=lambda item: item[1], reverse=True)[get_acc_count][1]
k_dec = sorted(dec_items, key=lambda item: item[1], reverse=True)[get_dec_count][1]

print("Getting acc and dec:")
print(get_acc_count, get_dec_count)
print(k_acc, k_dec)

for w in n_grams_total.keys():

    # Otherwise we would pick unique words.
    if w not in n_grams_acc or w not in n_grams_dec:
        continue

    # Considering bound to prevent from mistakenly annotated words.
    if n_grams_dec[w] < k_dec or n_grams_acc[w] < k_acc:
        continue

    r_acc = calc_pmi_for_terms(p_wc=n_grams_acc[w] / acc_total,
                               pw=n_grams_total[w] / total,
                               pc=acc_prob_total)

    r_dec = calc_pmi_for_terms(p_wc=n_grams_dec[w] / dec_total,
                               pw=n_grams_total[w] / total,
                               pc=dec_prob_total)

    so[w] = r_acc - r_dec
    print(so[w])

# Show the most acc. tokens and dec tokens.
max_acc = abs(max(so.values()))
max_dec = abs(min(so.values()))
for key in so.keys():
    if so[key] > 0:
        so[key] /= max_acc
    else:
        so[key] /= max_dec

# Show
items = list(so.items())
items_acc = sorted(items, key=lambda item: item[1], reverse=True)
items_dec = sorted(items, key=lambda item: item[1], reverse=False)
print(items_acc[:1000])
print(items_dec[:1000])


def reorder(ddf, texts, so):
    s_first_length = []
    total_removed = 0
    for t_ind, sentences in tqdm(enumerate(texts), "Reordering"):
        s_sos = []
        for s_ind, s in enumerate(sentences):
            s_terms = clean_sentence(s, do_pos=False).split()
            s_terms_so = [so[w] for w in s_terms if w in so]

            # We consider max/min as a metric for sentence.
            # depending on the actual sentence class.
            # Calculating most outlier sentences
            if len(s_terms_so) > 0:
                st_pos = [ts for ts in s_terms_so if ts > 0]
                st_neg = [ts for ts in s_terms_so if ts < 0]
                st_pos_len = len(st_pos) if len(st_pos) > 0 else 1
                st_neg_len = len(st_neg) if len(st_neg) > 0 else 1
                st_pos = sum(st_pos) if len(st_pos) > 0 else 0
                st_neg = sum(st_neg) if len(st_neg) > 0 else 0
                s_so = abs((st_pos/st_pos_len + st_neg/st_neg_len))
            else:
                s_so = 0

            s_sos.append((s_ind, s_so))

        s_sos_ordered = sorted(s_sos, key=lambda item: item[0], reverse=True)
        # remove the useless sentences.
        l_before = len(s_sos_ordered)
        s_sos_ordered = [(i, so) for i, so in s_sos_ordered if so is not 0]
        l_after = len(s_sos_ordered)
        total_removed += l_before - l_after

        ord_sentences = [sentences[i] for i, _ in s_sos_ordered]

        s_first_length.append(len(ord_sentences[0]))

        # update text.
        ddf.at[t_ind, "text"] = " ".join(ord_sentences)

    print("Avg sentences per doc. removed: {}".format(total_removed / len(texts)))

    # We also measure that first sentences in average are 150-205 tokens,
    # which is relatively large amount. So I think it might be reduced
    # because of the BERT model that fits only 512 tokens.
    print("Average statistics of the first: (max/min/avg)")
    print(max(s_first_length), min(s_first_length), np.mean(s_first_length))


reorder(df_train, [s for s, _ in texts], so)

# Reading test data.
test_c1 = reader.read(join(INPUT_DIR, "sample-orig-test-c1.tsv.gz"))
test_c2 = reader.read(join(INPUT_DIR, "sample-orig-test-c2.tsv.gz"))

# Apply the same for the test data of both tasks.
df_c1 = test_c1._df
reorder(df_c1, create_texts(df_c1), so)
df_c2 = test_c2._df
reorder(df_c2, create_texts(df_c2), so)

# Save output for the train and test data of both tasks.
df_train.to_csv("sample-train.csv", index=False)
df_c1.to_csv("c1_test.csv", index=False)
df_c2.to_csv("c2_test.csv", index=False)
