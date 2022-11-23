import os


# Import Data set
class Preprocessor:
    @staticmethod
    def corpus_importer(directory):
        entire_corpus = ""
        # iterate over files in  that directory
        for file_name in os.scandir(directory):
            # checking if it is a file
            if os.path.isfile(file_name):
                print(file_name)
                Text_file_Content = open(file_name, "r").read()
                try:
                    print(Text_file_Content)
                except:
                    print("Something didn't go well with this file: " + str(file_name))
        return entire_corpus

# Clean Data Set

# Break Data set into Question Answer pairs

# Try to break Question Answer pairs by topic

# Pick Topic to Explore

if __name__ == '__main__':
    a = Preprocessor.corpus_importer("../Datasets/Training/Extracted_from_wiki")
