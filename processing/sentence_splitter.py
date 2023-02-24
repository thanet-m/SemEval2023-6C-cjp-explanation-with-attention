import nltk


def split_text_on_sentences(text):
    origin_text = text.split()
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sents_bounds = [len(sent.split(' ')) for sent in tokenizer.tokenize(text)]

    # Split to regions.
    words_before = 0
    for s_ind, words_count in enumerate(sents_bounds):
        sents_bounds[s_ind] = (words_before, words_before + words_count)
        words_before += words_count

    sentences = []
    for b in sents_bounds:
        sentences.append(' '.join(origin_text[b[0]:b[1]]))

    return sentences
