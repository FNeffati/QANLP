import os
import re

import nltk


# Import Data set
def corpus_importer(directory):
    entire_corpus = ""
    # iterate over files in  that directory
    for file_name in os.scandir(directory):
        # checking if it is a file
        if os.path.isfile(file_name):
            Text_file_Content = open(file_name, "r").read()
            try:
                entire_corpus += Text_file_Content
            except:
                print("Something didn't go well with this file: " + str(file_name))
    return entire_corpus


def text_tokenizer(raw_corpus):
    corpus = re.sub(r'\n', '', raw_corpus)
    corpus = corpus.replace("[edit]", '')

    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    tokenized_corpus = tokenizer.tokenize(corpus.lower())
    return tokenized_corpus

# Break Data set into Question Answer pairs

# Try to break Question Answer pairs by topic

# Pick Topic to Explore


if __name__ == '__main__':
    a = corpus_importer("../Datasets/Training/Extracted_from_wiki")
    text_tokenizer(a)

