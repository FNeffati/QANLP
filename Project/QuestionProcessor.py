"""
In this class we'll be writing the rules of how each question is dealt with
"""
import math
import os

import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import deepdiff

stop_words = set(stopwords.words('english'))

embeddings_dict = {}

for file_name in os.scandir("../Datasets/Training/glove.6B"):
    if os.path.isfile(file_name):
        with open(file_name, 'r', encoding="utf−8") as f:
            try:
                for line in f:
                    values = line.split()
                    word = values[0]
                    vector = np.asarray(values[1:], "float32")
                    embeddings_dict[word] = vector
            except:
                print("Something didn't go well with this file: " + str(file_name))


def what_is_your_question():
    user_input = input("What would you like to know? ")
    return user_input


def most_similar_sentence(tokenized_file, question_sentence):
    best_matching_sentence_in_file = 0
    actual_sentence = ""
    important_words_from_question = process_question(question_sentence)
    for sentence in tokenized_file:
        important_words_from_sentence = process_sentence(sentence)
        if len(deepdiff.DeepDiff(important_words_from_question, important_words_from_sentence)) > 0:

            current_similarity = compare_sentence_with_question(sentence,question_sentence)
            if current_similarity > best_matching_sentence_in_file:
                best_matching_sentence_in_file = current_similarity
                actual_sentence = sentence
    print(actual_sentence)


"""
tried regular cosine similarity
not accurate at all. 
"""


def compare_sentence_with_question(sentence1, question):
    total_sentences_similarity = 0
    for word_1 in word_tokenize(sentence1):
        best_similarity_for_current_words = 0
        for word_2 in word_tokenize(question):
            try:
                vec1 = embeddings_dict[word_1]
                vec2 = embeddings_dict[word_2]
                current_similarity = cosine_similarity(vec1, vec2)
                if current_similarity > best_similarity_for_current_words:
                    best_similarity_for_current_words = current_similarity
                total_sentences_similarity += best_similarity_for_current_words
            except:
                continue
    return total_sentences_similarity


def process_sentence(sentence):

    #Can we preprocess our dataset so that we would not need to do the operations below for each sentence we have?

    stops = set(stopwords.words('english'))
    lowered = sentence.lower()
    sentence = word_tokenize(lowered)
    tagged = nltk.pos_tag(sentence)

    WDT_tags = ["NN", "NNP", "NNS", "JJ", "JJR", "JJS"]
    WP_tags = ["NN", "NNP", "NNS", "JJ", "JJR", "JJS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
    WRB_tags = ["NN", "NNP", "NNS", "JJ", "JJR", "JJS", "VB"]
    important = []
    """This is the first approach. 
    It first checks the type of question based on the first wh-question word, and then extracts the most relevant
    parts of speech.
    """
    # WDT words are: that what whatever which whichever
    if tagged[0][1] == "WDT":
        for item in tagged:
            if item[0] not in stops:
                if item[1] in WDT_tags:
                    important.append(item[0])
    # WP words are: that what whatever whatsoever which who whom whosoever
    elif tagged[0][1] == "WP":
        for item in tagged:
            if item[0] not in stops:
                if item[1] in WP_tags:
                    important.append(item[0])
    # WRB words are: how however whence whenever where whereby whereever wherein whereof why
    elif tagged[0][1] == "WRB":
        for item in tagged:
            if item[0] not in stops:
                if item[1] in WRB_tags:
                    important.append(item[0])

    '''The second approach just filters out stopwords and wh words'''
    """
    for item in tagged:
        if item[0] not in stops:
            if 'WDT' not in item[1] and 'WP' not in item[1] and 'WRB' not in item[1]:
                important.append(item[0])
    """
    # print("Your sentence seems to be about the following items:", str(important))
    return important


def process_question(user_q):
    # TODO Implement WH questions extraction
    stops = set(stopwords.words('english'))
    lowered = user_q.lower()
    question = word_tokenize(lowered)
    tagged = nltk.pos_tag(question)
    WDT_tags = ["NN", "NNP", "NNS", "JJ", "JJR", "JJS"]
    WP_tags = ["NN", "NNP", "NNS", "JJ", "JJR", "JJS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
    WRB_tags = ["NN", "NNP", "NNS", "JJ", "JJR", "JJS", "VB"]
    important = []
    """This is the first approach. 
    It first checks the type of question based on the first wh-question word, and then extracts the most relevant
    parts of speech.
    """
    #WDT words are: that what whatever which whichever
    if tagged[0][1] == "WDT":
        for item in tagged:
            if item[0] not in stops:
                if item[1] in WDT_tags:
                    important.append(item[0])
    #WP words are: that what whatever whatsoever which who whom whosoever
    elif tagged[0][1] == "WP":
        for item in tagged:
            if item[0] not in stops:
                if item[1] in WP_tags:
                    important.append(item[0])
    # WRB words are: how however whence whenever where whereby whereever wherein whereof why
    elif tagged[0][1] == "WRB":
        for item in tagged:
            if item[0] not in stops:
                if item[1] in WRB_tags:
                    important.append(item[0])

    '''The second approach just filters out stopwords and wh words'''
    """
    for item in tagged:
        if item[0] not in stops:
            if 'WDT' not in item[1] and 'WP' not in item[1] and 'WRB' not in item[1]:
                important.append(item[0])
    """
    # print("Your question seems to be about the following items:", str(important))
    return important


def cosine_similarity(wvec1, wvec2):
    numerator, denom1, denom2 = 0, 0, 0
    for item in range(len(wvec1)):
        numerator += wvec1[item] * wvec2[item]
        denom1 += wvec1[item] * wvec1[item]
        denom2 += wvec2[item] * wvec2[item]
    similarity = numerator / (math.sqrt(denom1 * denom2))

    return similarity


if __name__ == '__main__':
    user_question = what_is_your_question()
    test_file = ["Hello, my name is james.", "I am 53 years old.", "james is British"]
    most_similar_sentence(test_file, user_question)
