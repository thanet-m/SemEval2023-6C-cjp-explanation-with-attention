
list_of_words = ["and", "the", "not", "other", "all", "turn", "certainly", "percent"]

VOCAB = ["the", "of", "he", "was", "be", "and", "to", "in", "this", "that",
         "we", "have", "had", "been", "were", "any", "other", "order", "a", "where",
         "incidence", "process", "demolish", "therefore", "mentioned", "prescribed",
         "qualifications", "necessary", "subsection", "became", "scientists", "suggestions",
         "members", "service", "allowance", "as", "well", "with", "tax", "clause", "case",
         "writ", "into", "judge", "by", "these", "act", "for", "dated", "which", "they", "at",
         "all", "his", "west", "east", "north", "south", "no", "even", "if", "there", "more",
         "less", "than", "are", "who", "public", "appeal", "before", "after", "us", "civil",
         "government", "set", "out", "up", "down", "far", "so", "but", "also",
         "it", "is", "does", "an", "on", "under", "high", "or", "state", "number"]


def fix_words(text):

    # Replace weird breaks.
    for w in list_of_words:
        for i in range(len(w) - 1):
            nw = " {}\n{} ".format(w[:i + 1], w[i + 1:])
            text = text.replace(nw, w)

    for w in VOCAB:
        text = text.replace(" {}\n".format(w), " {} ".format(w))

    return text
