"""
This class is responsible for extracting 20 random Wikipedia pages related to a provided topic.
When code is executed, the program will output a file called "Corpus.txt" in the Training/Extracted_from_wiki directory
that contains all the text of those documents.

Author: Fehmi Neffati
"""
from bs4 import BeautifulSoup
import requests
import re
import random


class WikiExtractor:
    @staticmethod
    def url_extractor(desired_topic_wiki_url):
        """
        function to extract multiple wikipedia articles related to the starting point url
        :param desired_topic_wiki_url: Starting point url
        :return: a list of urls of related wiki pages
        """
        doc = BeautifulSoup(requests.get(desired_topic_wiki_url).text, "html.parser")
        tags = doc.findAll("a")
        links = []
        unwanted = ["/wiki/File", "/wiki/Category", "/wiki/Wiki", "/wiki/Help", "/wiki/Special", "/wiki/Portal",
                    "/wiki/Wikipedia", "/wiki/Doi_", "/wiki/PMID_", "/wiki/PMC_", "/wiki/S2CID_", "/wiki/Main_Page"]
        for tag in tags:
            current = re.findall(r"href=\W+/[Ww]iki/\w+", str(tag))
            if len(current) > 0:
                if current[0][6:] not in unwanted:
                    links.append(current)
        wanted = []
        for link in links:
            wanted.append("https://en.wikipedia.org" + link[0][6:])

        wanted = random.choices(wanted, k=20)
        wanted.append(desired_topic_wiki_url)
        print(wanted)
        return wanted

    @staticmethod
    def text_exporter(random_url_list):
        """
        This method should extract all the text from the scraped web paged and export it to a
        :param random_url_list: The list of randomly picked URLs from the @url_extractor method
        :return: Void. The method writes a .txt file with the contents of the web pages
        """
        try:
            with open("../Datasets/Training/Extracted_from_wiki/Corpus" + '.txt', 'w') as f:
                for link in random_url_list:
                    doc = BeautifulSoup(requests.get(link).text, "html.parser")
                    for line in doc:
                        f.write(str(line.text))
                print("Text extraction process complete.")
        except:
            print("Something went wrong with Text extraction process, please debug.")


if __name__ == '__main__':
    target_topic_url = "https://en.wikipedia.org/wiki/History_of_the_United_States"
    url_list = WikiExtractor.url_extractor(target_topic_url)
    WikiExtractor.text_exporter(url_list)

