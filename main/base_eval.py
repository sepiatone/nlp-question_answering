"""
Implementations of the bm25 and tf-idf algorithms to compute baseline evaluation metrics for the Android dataset
"""

from collections import Counter, defaultdict
import math
import numpy as np
import util
from util import AUCMeter


class BM25_TFIDF:    
    """ 
    Compute the idf (inverse document frequency) statistic for a term (word) using the formula for bm (BMIDF) and the usual formula for tf-idf (TFIDF)
    Here each line in the stackexchange android dataset consisiting of an index, a title (question) and body is a "document"
    
    The idf statistic can be used to compute the bm25 score and tf-idf score of a dataset
    """
    
    def __init__(self, data_file, delimiter = '\t'):
        self.doc_term_counter = defaultdict(lambda: 0)  # the number of documents in which a term is present
        self.documents = dict() # dict of documents with values being a dict of terms and values of the number of their occurences
        self.document_lengths = dict() # the number of words in each document
        self.total_lengths = 0 # total number of words in all documents
        self.N = 0 # number of documents
        
        with open(data_file) as data:
            
            for line in data:
                idx, title, body = line.split(delimiter)
                
                if len(title) == 0:
                    continue
                
                title = title.strip().split()
                body = body.strip().split()
                text = title + body
                self.documents[idx] = Counter(text)
                
                for term in self.documents[idx]:
                    self.doc_term_counter[term] += 1
                
                self.document_lengths[idx] = len(text)
                self.total_lengths += len(text)
                self.N += 1
        
        self.avg_len = self.total_lengths / float(self.N)
        self.BMIDF = dict()
        self.TFIDF = dict()
        
        for term in self.doc_term_counter:
            self.BMIDF[term] = math.log((self.N - self.doc_term_counter[term] + 0.5) / (self.doc_term_counter[term] + 0.5))
            self.TFIDF[term] = math.log((self.N)/(self.doc_term_counter[term] + 1)) + 1

    """
    Commpute the bm25 score
    """
    def BM25Score(self, q1, q2s, k1 = 1.5, b = 0.75):        
        scores = []
        
        for q2 in q2s:
            doc = self.documents[q2]
            commonTerms = set(self.documents[q1]) & set(doc)
            tmp_score = []
            doc_terms_len = self.document_lengths[q2]
            for term in commonTerms:
                upper = (doc[term] * (k1+1))
                below = ((doc[term]) + k1*(1 - b + b*doc_terms_len/self.avg_len))
                tmp_score.append(self.BMIDF[term] * upper / below)
            scores.append(sum(tmp_score))
        return np.array(scores)

    """
    Compute the tf-idf score
    """
    def TFIDFScore(self, q1, q2s):
        scores = []
        for q2 in q2s:
            doc = self.documents[q2]
            commonTerms = set(self.documents[q1]) & set(doc)
            tmp_score = []
            doc_terms_len = self.document_lengths[q2]
            for term in commonTerms:
                tmp_score.append( self.TFIDF[term] * math.sqrt(doc[term]) * 1.0/math.sqrt(doc_terms_len))
            scores.append(sum(tmp_score))
        return np.array(scores)
    
    
def evaluate_BM25_AUC(data, model):
    AUC = AUCMeter()
    AUC.reset()
    
    for question, possibilities, labels in data:
        labels = np.array(labels)
        scores = model.BM25Score(question, possibilities)
        assert len(scores) == len(labels)
        AUC.add(scores, labels)
    
    return AUC.value(0.05)


def evaluate_TFIDF_AUC(data, model):
    AUC = AUCMeter()
    AUC.reset()
    
    for question, possibilities, labels in data:
        labels = np.array(labels)
        scores = model.TFIDFScore(question, possibilities)
        assert len(scores) == len(labels)
        AUC.add(scores, labels)
    
    return AUC.value(0.05)


corpus = "../data/stackexchange_android/corpus.txt"
pos_dev = "../data/stackexchange_android/dev.pos.txt"
neg_dev = "../data/stackexchange_android/dev.neg.txt"
pos_test = "../data/stackexchange_android/test.pos.txt"
neg_test = "../data/stackexchange_android/test.neg.txt"

def main():
    dev_annotations = util.read_annotations_2(pos_dev, neg_dev, -1, -1)
    test_annotations = util.read_annotations_2(pos_test, neg_test, -1, -1)    
    
    model = BM25_TFIDF(corpus)

    print("The BM25 AUC for the dev dataset is {}".format(evaluate_BM25_AUC(dev_annotations, model)))
    print("The TD-IDF AUC for the dev dataset is {}".format(evaluate_TFIDF_AUC(dev_annotations, model)))    
    
    print("The BM25 AUC for the test dataset is {}".format(evaluate_BM25_AUC(test_annotations, model)))
    print("The TD-IDF AUC for the test dataset is {}".format(evaluate_TFIDF_AUC(test_annotations, model)))

if __name__ == "__main__":    
    main()
else:
    print("run the file directly to get the baseline evaluation metrics for the android dataset")