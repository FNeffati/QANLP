import Preprocessor
import QuestionProcessor


def prep():
    corpus = Preprocessor.corpus_importer("../Datasets/Training/Extracted_from_wiki")
    token_corpus = Preprocessor.text_tokenizer(corpus)
    return token_corpus


if __name__ == '__main__':
    print("\nStarting tokenization: ")
    tokenized_corpus = prep()
    print("\nFinished tokenization: ")
    print("\nStarting QUESTION ANSWERING: ")
    user_question = QuestionProcessor.what_is_your_question()
    print("Your question is: ", user_question)
    QuestionProcessor.most_similar_sentence(tokenized_corpus, user_question)
