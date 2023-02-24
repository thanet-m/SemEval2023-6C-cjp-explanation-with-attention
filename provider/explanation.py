import nltk.data
import numpy as np
from os.path import join, exists
from zipfile import ZipFile
from arenets.arekit.common.data.input.reader import BaseReader
from arenets.arekit.common.data_type import DataType
from utils import VocabRepositoryUtils, NpzRepositoryUtils


def crop_sentence_optionally(terms, attention, window_size):
    assert(isinstance(terms, list))
    assert(isinstance(window_size, int))

    if len(terms) < window_size:
        return terms

    begin = 0
    max_avg_att = 0
    for i in range(len(terms)-window_size):
        avg_att = np.mean(attention[i:i+window_size])
        if avg_att > max_avg_att:
            begin = i
            max_avg_att = avg_att

    return terms[begin:begin+window_size]


def generate_windowed_explanation(att_sent_avg, sents_bounds, output_bound,
                                  origin_text_terms, attention_text, sentence_window):
    exp = []

    for s_ind, _ in att_sent_avg:
        if len(exp) > output_bound:
            break
        b = sents_bounds[s_ind]
        cropped = crop_sentence_optionally(terms=origin_text_terms[b[0]:b[1]],
                                           attention=attention_text[b[0]:b[1]],
                                           window_size=sentence_window)
        exp.extend(cropped)

    return ' '.join(exp)


def provide_explanation(model_name, input_dir, input_terms_count, sample_type, output_dir,
                        extention, reader, sentence_window=10, output_bound=64):
    """ Core function responsible for the explanation generation.
        In the case of Attention-CNN approach.
    """
    assert(isinstance(reader, BaseReader))

    subdir = "{}/hidden/".format(model_name)

    samples_to_str = {
        DataType.Dev: "dev",
        DataType.Test: "test"
    }

    # Preparing paths.
    y_data_filepath = join(input_dir, subdir, "idparams_y_labels-{}.npz".format(sample_type))
    att_data_filepath = join(input_dir, subdir, "idparams_ATT_Weights-{}.npz".format(sample_type))
    x_data_ids_filepath = join(input_dir, subdir, "idparams_x-{}.ids.npz".format(sample_type))
    y_data_ids_filepath = join(input_dir, subdir, "idparams_y_labels-{}.ids.npz".format(sample_type))

    print(y_data_filepath)
    print(att_data_filepath)
    if not (exists(y_data_filepath) and exists(att_data_filepath)):
        # Leaving, because we missed at least one data required to perform explanation.
        print("LEAVING")
        return

    print("Explaining for model: {} ({})".format(model_name, input_terms_count))

    # Loading data.
    y_data = NpzRepositoryUtils.load(y_data_filepath)
    att_data = NpzRepositoryUtils.load(att_data_filepath)
    x_data_ids = NpzRepositoryUtils.load(x_data_ids_filepath)
    y_data_ids = NpzRepositoryUtils.load(y_data_ids_filepath)

    # Reading the original data.
    etalon_data = {}
    for _, row in reader.read(join(input_dir, "sample-{}.{}".format(samples_to_str[sample_type], extention))):
        row_id = row["id"]
        text = row["text"].split()[:input_terms_count]
        etalon_data[row_id] = text

    # Extracting required data from vocabulary.
    vocab = VocabRepositoryUtils.load(join(input_dir, "vocab.txt"))
    v = [""] * (len(vocab) + 1)
    for value, ind in vocab:
        v[int(ind)] = value

    # Do summary.
    explanations = {}
    for text_ind in range(len(y_data)):
        assert (y_data_ids[text_ind] == x_data_ids[text_ind])

        # Picking the related text from the original file.
        label = y_data[text_ind]
        x_data_ind = x_data_ids[text_ind]
        origin_text_terms = etalon_data[x_data_ind]
        attention_text = np.squeeze(att_data[text_ind])

        # We consider all examples but not duplicate them.
        # Since everything was formed in batches we may experience with repetitions.
        if x_data_ind in explanations:
            continue

        # Do tokenization.
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        sents_bounds = [len(sent.split(' ')) for sent in tokenizer.tokenize(' '.join(origin_text_terms))]

        # Split to regions.
        words_before = 0
        for s_ind, words_count in enumerate(sents_bounds):
            sents_bounds[s_ind] = (words_before, words_before + words_count)
            words_before += words_count

        # extract avg sent attentions.
        att_sent_avg = [(s_ind, np.mean(attention_text[s_bounds[0]:s_bounds[1]]))
                        for s_ind, s_bounds in enumerate(sents_bounds)]
        att_sent_avg = sorted(att_sent_avg, key=lambda item: item[1], reverse=True)

        # Select k-terms most important sentences.
        exp = generate_windowed_explanation(att_sent_avg=att_sent_avg,
                                            sents_bounds=sents_bounds,
                                            output_bound=output_bound,
                                            origin_text_terms=origin_text_terms,
                                            attention_text=attention_text,
                                            sentence_window=sentence_window)

        # Save.
        explanations[x_data_ind] = {
            "text": exp,
            "label": int(label)
        }

    # Saving result. This is a hard-coded format for Codalab competition.
    target = "{output_dir}/explanations-{inp_size}-{model_name}-{sample_type}.zip".format(
        output_dir=output_dir, inp_size=input_terms_count, model_name=model_name, sample_type=sample_type)
    print("Saving: {}".format(target))

    with ZipFile(target, 'w') as myzip:
        contents = ["uid,decision,explanation"]
        for exp_id, explanation in explanations.items():
            line_content = [exp_id,
                            "Accepted" if explanation["label"] == 1 else "Denied",
                            "\"{}\"".format(explanation["text"].replace(',', ''))]
            contents.append(",".join(line_content))

        myzip.writestr("predictions.csv", "\n".join(contents))
