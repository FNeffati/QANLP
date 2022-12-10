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
import json

stop_words = set(stopwords.words('english'))

embeddings_dict = {}





def make_qid_to_has_ans(dataset):
  qid_to_has_ans = {}

  for article in dataset:

    for p in article['paragraphs']:
      for qa in p['qas']:
        qid_to_has_ans[qa['id']] = bool(qa['answers'])
  return qid_to_has_ans

def extract_qas(dataset):
  qas = {}

  for article in dataset:

    for p in article['paragraphs']:
      for qa in p['qas']:
        answer = json.loads(json.dumps(qa["answers"]))
        #print(answer[0])
        #tup = (qa["question"],qa["answers"]["text"])
        if(len(answer) > 0):

            tup = (qa["question"], answer[0]["text"])
            qas[qa["id"]] = tup
  return qas

def getGloveVectors():

    for file_name in os.scandir("Datasets/Training/glove"):
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
    print("Done getting vectors")

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
    else:
        for item in tagged:
            if item[0] not in stops:
                if 'WDT' not in item[1] and 'WP' not in item[1] and 'WRB' not in item[1]:
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

def get_qas():
    with open("Datasets/Training/train-v2.txt", 'r') as f:
        json_data = json.load(f)
        data = json_data['data']

        print("Done getting json data.")
        #print(data[0]["paragraphs"])
        qas = extract_qas(data)

        #print(list(qas.items())[0][1][0])

        #print(list(test_data.items())[0:10])
        return qas


def simple_qa_vectors(data, vector_dict):
    qa_vectors = {}
    empty = np.zeros(shape=(50,))
    test = True
    items = list(data.items())
    for entry in items:
        question_vectors = np.zeros(shape=(50,))
        answer_vectors = np.zeros(shape=(50,))
        question = word_tokenize(entry[1][0])
        answer = word_tokenize(entry[1][1])
        try:

            for word in question:
                word = word.lower()
                question_vectors += vector_dict[word]


        except:

            question_vectors += empty

        try:
            for word in answer:
                word = word.lower()
                answer_vectors += vector_dict[word]
        except:

            answer_vectors += empty

        qa_vectors[entry[0]] = (question_vectors, answer_vectors)

    print("done finding vectors")
    return qa_vectors


def get_arrays(vectors):
    data = list(vectors.items())


    q_train = []
    a_train = []


    for entry in data:
        q_train.append(entry[1][0])
        a_train.append(entry[1][1])





    return q_train, a_train

def init_params():
    W1 = np.random.rand(50, 50) - 0.5
    b1 = np.random.rand(50, 1) - 0.5
    #W2 = np.random.rand(10, 50) - 0.5
    #b2 = np.random.rand(50, 1) - 0.5
    return W1, b1
def ReLU(Z):
    return np.maximum(Z, 0)


def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A


def forward_prop(W1, b1, X):
    Z1 = W1.dot(X) + b1
    A1 = softmax(Z1)

    return Z1, A1


def ReLU_deriv(Z):
    return Z > 0


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y, m):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, dW1, db1, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    return W1, b1


def get_predictions(A):
    return np.argmax(A, 0)


def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


def gradient_descent(X, Y, alpha, iterations, n):
    W1, b1 = init_params()
    for i in range(iterations):
        Z1, A1 = forward_prop(W1, b1, X)
        dW1, db1 = backward_prop(Z1, A1, W1, X, Y, n)
        W1, b1 = update_params(W1, b1, dW1, db1, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A1)
            print(get_accuracy(predictions, Y))
    return W1, b1





def make_predictions(X, W1, b1):
    _, A1 = forward_prop(W1, b1, X)
    predictions = get_predictions(A1)
    return predictions


if __name__ == '__main__':

    qas = get_qas()
    getGloveVectors()
    qa_vectors = simple_qa_vectors(qas, embeddings_dict)


    q_train, a_train = get_arrays(qa_vectors)
    q_train = np.array(q_train)
    a_train = np.array(a_train)
    q_train = q_train.T
    a_train = a_train.T
    n = q_train.shape[1]
    W1, b1 = gradient_descent(q_train, a_train, 0.10, 100, n)

    #W1, b1, W2, b2 = init_params()
    #Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, q_train)
    #print(Z1)

    """
    
    qa_vectors_array = list(qa_vectors.items())
    print(qa_vectors_array[0])
    np.random.shuffle(qa_vectors_array)
    print(qa_vectors_array[0])
    """
    #print(qa_vectors["56be85543aeaaa14008c9063"][1])
    #print(list(qa_vectors.items())[1])
    #user_question = what_is_your_question()
    #test_file = ["Hello, my name is james.", "I am 53 years old.", "james is British"]
    #most_similar_sentence(test_file, user_question)

