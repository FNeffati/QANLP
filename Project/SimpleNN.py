"""
This code was developed by Fehmi Neffati and Vlad Tsimoshchanka using the code of Samson Zhang -
https://www.kaggle.com/wwsalmon/simple-mnist-nn-from-scratch-numpy-no-tf-keras
"""
import math
import os
import traceback

import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import deepdiff
import json

stop_words = set(stopwords.words('english'))

#Dictionary for word vectors
embeddings_dict = {}

#Extract questions and
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
                # print(answer[0])
                # tup = (qa["question"],qa["answers"]["text"])
                if (len(answer) > 0):
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

            current_similarity = compare_sentence_with_question(sentence, question_sentence)
            if current_similarity > best_matching_sentence_in_file:
                best_matching_sentence_in_file = current_similarity
                actual_sentence = sentence
    print(actual_sentence)


"""
tried regular cosine similarity
not accurate at all. 
"""

#Comparing two sentences which one is more relevant to a question
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

#Extract most important things
def process_sentence(sentence):
    # Can we preprocess our dataset so that we would not need to do the operations below for each sentence we have?

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

#Extract most important things
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
    # print("Your question seems to be about the following items:", str(important))
    return important


def cosine_similarity(wvec1, wvec2):
    numerator, denom1, denom2 = 0, 0, 0
    for item in range(len(wvec1)):
        numerator += wvec1[item] * wvec2[item]
        denom1 += wvec1[item] * wvec1[item]
        denom2 += wvec2[item] * wvec2[item]
    try:
        similarity = numerator / (math.sqrt(denom1 * denom2))
    except:
        similarity = 0.001

    return similarity

#Code from https://www.kaggle.com/code/cdabakoglu/word-vectors-cosine-similarity/notebook
def cosine_similarity2(a, b):
    nominator = np.dot(a, b)

    a_norm = np.sqrt(np.sum(a ** 2))
    b_norm = np.sqrt(np.sum(b ** 2))

    denominator = a_norm * b_norm

    cosine_similarity = nominator / denominator

    return cosine_similarity

#Extract qustions and answers from dataset
def get_qas():
    with open("Datasets/Training/train-v2.txt", 'r') as f:
        json_data = json.load(f)
        data = json_data['data']

        print("Done getting json data.")
        # print(data[0]["paragraphs"])
        qas = extract_qas(data)

        # print(list(qas.items())[0][1][0])

        # print(list(test_data.items())[0:10])
        return qas

#Create word vectors for questions and answers
def simple_qa_vectors(data, vector_dict):
    qa_vectors = {}
    reverse_lookup = {}
    answer_list = []
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
        #print("HEHEHE")
        #print(type(question_vectors))
        #print(type(type(question_vectors)))
        reverse_lookup[tuple(question_vectors)] = answer
        answer_list.append(entry[1][1])

    print("done finding vectors")
    return qa_vectors, reverse_lookup, answer_list

#Get answers and questions as separate arrays
def get_arrays(vectors):
    data = list(vectors.items())

    q_train = []
    a_train = []

    for entry in data:
        q_train.append(entry[1][0])
        a_train.append(entry[1][1])


    return q_train, a_train

#Set up weights and bias
def init_params():
    W1 = np.random.rand(1, 50) - 0.5
    b1 = np.random.rand(1, 1) - 0.5
    # W2 = np.random.rand(10, 50) - 0.5
    # b2 = np.random.rand(50, 1) - 0.5
    return W1, b1




def sigmoid(x):  # logistic function
    return 1 / (1 + np.exp(-x))


def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))  # loss function


def forward_prop(W1, b1, X):
    Z1 = W1.dot(X) + b1
    A1 = sigmoid(Z1)

    return Z1, A1





# We go backwards the neural network to see the errors
def backward_prop(Z1, A1, W1, X, Y, m):
    error = A1 - Y
    dZ1 = W1.dot(error) * sigmoid_der(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1

#We rectify the weights and bias here
def update_params(W1, b1, dW1, db1, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    return W1, b1

'''
def get_predictions(A):
    return np.argmax(A, 0)


def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size
'''

def gradient_descent(X, Y, alpha, iterations, n):
    W1, b1 = init_params()
    for i in range(iterations):
        Z1, A1 = forward_prop(W1, b1, X)
        dW1, db1 = backward_prop(Z1, A1, W1, X, Y, n)
        W1, b1 = update_params(W1, b1, dW1, db1, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            #print(W1.shape)
            #print(X.shape)

    return W1, b1

'''
def make_predictions(X, W1, b1):
    _, A1 = forward_prop(W1, b1, X)
    predictions = get_predictions(A1)
    return predictions
'''
#Get nn results for a user question
def process_user_question(question, weights, bias, vector_dict):
    tokens = word_tokenize(question)
    empty = np.zeros(shape=(50,))
    question_vectors = np.zeros(shape=(50,))
    #get vectors for question
    try:

        for word in tokens:
            word = word.lower()
            question_vectors += vector_dict[word]


    except:

        question_vectors += empty
    #Multiply each question vector by respective wieght and apply bias
    for i in range(50):
        question_vectors[i] *= weights[0][i]
        question_vectors[i] += bias
    #pass the processed question vectors to activation function
    result = sigmoid(question_vectors)
    #returns a (50,) array of floats which are the results of this question after activation
    """
    Example:
        [0.56036192 0.44065425 0.56008441 0.6528372  0.65577416 0.59644015
         0.33923938 0.58573419 0.5695277  0.52240839 0.57544139 0.37500304
         0.46981236 0.49676253 0.52928405 0.61372548 0.38301962 0.63709157
         0.54428967 0.39100132 0.59976918 0.48501537 0.40999988 0.45127281
         0.50799247 0.95919028 0.40104783 0.59294465 0.81835053 0.71571873
         0.99765124 0.73533718 0.61152855 0.57363327 0.52270125 0.5386597
         0.52283112 0.51487598 0.43436114 0.53439491 0.65162222 0.46008518
         0.57393716 0.66137629 0.47499859 0.54758152 0.50071964 0.46848847
         0.45128081 0.46788413]
    (50,)
    """
    return(result)

def compare_questions(user_question, dataet_question):
    total_sentences_similarity = 0
    for word_1 in word_tokenize(dataet_question):
        best_similarity_for_current_words = 0
        for word_2 in word_tokenize(user_question):
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



if __name__ == '__main__':
    qas = get_qas()
    getGloveVectors()
    qa_vectors, reverse_lookup , answer_list= simple_qa_vectors(qas, embeddings_dict)

    q_train, a_train = get_arrays(qa_vectors)
    q_train = np.array(q_train)
    qsts = q_train
    a_train = np.array(a_train)
    q_train = q_train.T
    a_train = a_train.T
    n = q_train.shape[1]
    W1, b1 = gradient_descent(q_train, a_train, 0.10, 400, n)
    #print(W1)
    #print(W1[0][0])

    user_question = input("Ask a question: ")
    processed_question = process_user_question(user_question, W1, b1, embeddings_dict)
    best_sim = 0
    best_sentence = []

    a_train = a_train.T
    count = 0
    #test = True
    for answer in a_train:
        '''
        if(test):
            sim = cosine_similarity(answer, processed_question)
            print(answer_list[count])
            print(sim)
            test = False
        '''
        current = cosine_similarity(answer, processed_question)
        if current > best_sim:
            try:
                best_sentence = answer_list[count]
                best_sim = current
            except:
                continue
        count += 1

    print(best_sentence)
    print(best_sim)

    #print(processed_question)
    #print(processed_question.shape)

    # W1, b1, W2, b2 = init_params()
    # Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, q_train)
    # print(Z1)

    """

    qa_vectors_array = list(qa_vectors.items())
    print(qa_vectors_array[0])
    np.random.shuffle(qa_vectors_array)
    print(qa_vectors_array[0])
    """
    # print(qa_vectors["56be85543aeaaa14008c9063"][1])
    # print(list(qa_vectors.items())[1])
    # user_question = what_is_your_question()
    # test_file = ["Hello, my name is james.", "I am 53 years old.", "james is British"]
    # most_similar_sentence(test_file, user_question)

