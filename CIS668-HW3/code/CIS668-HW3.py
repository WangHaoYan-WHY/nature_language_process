from nltk.corpus import sentence_polarity
import random
import nltk
from nltk import FreqDist
import nltk.data
import pandas as pd
from nltk.metrics import ConfusionMatrix
import re


# return true if it is non-alpha
def alpha_filter(w):
    # pattern to match a word of non-alphabetical characters
    pattern = re.compile('^[^a-z]+$')
    if pattern.match(w):
        return True
    else:
        return False

# do tokenization
def do_word_tokenize(content):
    # do word tokenizing process
    content = content.replace("\r\n", " ")
    content = content.replace("\r", " ")
    content = content.replace("\n", " ")
    content = content.replace("@", " ")
    tokens = nltk.word_tokenize(content)
    return tokens


# do lower the words
def do_lower(content):
    # set all words as lowercase
    words = [w.lower() for w in content]
    return words


# print out the top 50 words
def do_print_top_50(content):
    # print top 50 frequency
    freq_dist = FreqDist(content)
    top_keys = freq_dist.most_common(50)
    for pair in top_keys:
        print(pair)
    print('-----------------------------------')


# preprocess the review data
def preprocess_data(onePartContent):
    # do word tokenize of the raw content
    tokens_content = do_word_tokenize(onePartContent)
    # set all the words in the content to be as lowercase
    lower_words = do_lower(tokens_content)
    return lower_words


# split the text into sentences
def split_sentences(content):
    sentences = nltk.sent_tokenize(content)
    return sentences


# get the sentence, category pairs
def get_document():
    documents = [(sent, category) for category in sentence_polarity.categories() for sent in
                 sentence_polarity.sents(categories=category)]
    random.shuffle(documents)
    return documents


# get set of words for features
def get_word_features(documents):
    all_words_list = [word for (sent, category) in documents for word in sent]
    all_words = nltk.FreqDist(all_words_list)
    word_items = all_words.most_common(2000)
    word_features = [word for (word, freq) in word_items]
    return word_features


# get unigram feature
def document_features_contain(document, word_features):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features


# use the unigram feature to train the classifier
def document_features_contain_train(documents, word_features):
    featuresets = [(document_features_contain(d, word_features), c) for (d, c) in documents]
    train_set, test_set = featuresets[1000:], featuresets[:1000]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print (nltk.classify.accuracy(classifier, test_set))
    return test_set, classifier


# creates a Subjectivity Lexicon
def readSubjectivity(path):
    flexicon = open(path, 'r')
    # initialize an empty dictionary
    sldict = {}
    for line in flexicon:
        fields = line.split()   # default is to split on whitespace
        # split each field on the '=' and keep the second part as the value
        strength = fields[0].split("=")[1]
        word = fields[2].split("=")[1]
        posTag = fields[3].split("=")[1]
        stemmed = fields[4].split("=")[1]
        polarity = fields[5].split("=")[1]
        if (stemmed == 'y'):
            isStemmed = True
        else:
            isStemmed = False
        # put a dictionary entry with the word as the keyword
        #     and a list of the other values
        sldict[word] = [strength, posTag, isStemmed, polarity]
    return sldict


# get SL features
def SL_features(document, word_features, SL):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    weakPos = 0
    strongPos = 0
    weakNeg = 0
    strongNeg = 0
    for word in document_words:
        if word in SL:
            strength, posTag, isStemmed, polarity = SL[word]
            if strength == 'weaksubj' and polarity == 'positive':
                weakPos += 1
            if strength == 'strongsubj' and polarity == 'positive':
                strongPos += 1
            if strength == 'weaksubj' and polarity == 'negative':
                weakNeg += 1
            if strength == 'strongsubj' and polarity == 'negative':
                strongNeg += 1
            features['positivecount'] = weakPos + (2 * strongPos)
            features['negativecount'] = weakNeg + (2 * strongNeg)
    return features


# use SL features to train
def SL_features_train(documents, word_features, SL):
    SL_featuresets = [(SL_features(d, word_features, SL), c) for (d, c) in documents]
    train_set, test_set = SL_featuresets[1000:], SL_featuresets[:1000]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print(nltk.classify.accuracy(classifier, test_set))
    return test_set, classifier


# get SL features
def new_SL_features(document, word_features, SL):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    weakPos = 0
    strongPos = 0
    weakNeg = 0
    strongNeg = 0
    for word in document_words:
        if word in SL:
            strength, posTag, isStemmed, polarity = SL[word]
            if strength == 'weaksubj' and polarity == 'positive':
                weakPos += 1
            if strength == 'strongsubj' and polarity == 'positive':
                strongPos += 1
            if strength == 'weaksubj' and polarity == 'negative':
                weakNeg += 1
            if strength == 'strongsubj' and polarity == 'negative':
                strongNeg += 1
            features['wpositivecount'] = weakPos
            features['sposativecount'] = strongPos
            features['wnegitivecount'] = weakNeg
            features['snegativecount'] = strongNeg
    return features


# use SL features to train
def new_SL_features_train(documents, word_features, SL):
    SL_featuresets = [(new_SL_features(d, word_features, SL), c) for (d, c) in documents]
    train_set, test_set = SL_featuresets[1000:], SL_featuresets[:1000]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print(nltk.classify.accuracy(classifier, test_set))
    return test_set, classifier


# negation features
def NOT_features(document, word_features, negationwords):
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = False
        features['contains(NOT{})'.format(word)] = False  # go through document words in order
    for i in range(0, len(document)):
        word = document[i]
        if ((i + 1) < len(document)) and ((word in negationwords) or (word.endswith("n't"))):
            i += 1
            features['contains(NOT{})'.format(document[i])] = (document[i] in word_features)
        else:
            features['contains({})'.format(word)] = (word in word_features)
    return features


# use negation features to train
def NOT_features_train(documents, word_features, negationwords):
    NOT_featuresets = [(NOT_features(d, word_features, negationwords), c) for (d, c) in documents]
    train_set, test_set = NOT_featuresets[200:], NOT_featuresets[:200]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print(nltk.classify.accuracy(classifier, test_set))
    return test_set, classifier


# filter stop words's words features
def filter_stop_words(documents):
    all_words_list = [word for (sent, category) in documents for word in sent]
    stopwords = nltk.corpus.stopwords.words('english')
    # all_words_list = [w for w in all_words_list if not alpha_filter(w)]
    newstopwords = [word for word in stopwords if word not in new_list]
    new_all_words_list = [word for word in all_words_list if word not in newstopwords]
    new_all_words = nltk.FreqDist(new_all_words_list)
    new_word_items = new_all_words.most_common(2000)
    new_word_features = [word for (word, count) in new_word_items]
    return new_word_features


# print the precision, recall and F-measure scores
def printmeasures(label, refset, testset):
    print(label, 'precision:', nltk.precision(refset, testset))
    print(label, 'recall:', nltk.recall(refset, testset))
    print(label, 'F-measure:', nltk.f_measure(refset, testset))


# analyze the precision, recall and F-measure score
def analyze_measures(test_set, classifier):
    reflist = []
    testlist = []
    for (features, label) in test_set:
        reflist.append(label)
        testlist.append(classifier.classify(features))
    cm = ConfusionMatrix(reflist, testlist)
    refneg = set([i for i, label in enumerate(reflist) if label == 'neg'])
    refpos = set([i for i, label in enumerate(reflist) if label == 'pos'])
    testneg = set([i for i, label in enumerate(testlist) if label == 'neg'])
    testpos = set([i for i, label in enumerate(testlist) if label == 'pos'])
    print("For pos")
    printmeasures('pos', refpos, testpos)
    print("For neg")
    printmeasures('neg', refneg, testneg)


negationwords = ['no', 'not', 'never', 'none', 'nowhere', 'nothing', 'noone', 'rather', 'hardly', 'scarcely', 'rarely', 'seldom', 'neither', 'nor']
reservewords = ['again', 'more', 'most', 'only', 'too', 'very']
notmeaningwords = ["don", "don't",'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
new_list = negationwords + reservewords + notmeaningwords

'''
SLpath = 'subjclueslen1-HLTEMNLP05.tff'
documents = get_document()
word_features = filter_stop_words(documents)
SL = readSubjectivity(SLpath)
test_set, classifier = SL_features_train(documents, word_features, SL)
analyze_measures(test_set, classifier)
'''

'''
documents = get_document()
word_features = filter_stop_words(documents)
test_set, classifier = document_features_contain_train(documents, word_features)
analyze_measures(test_set, classifier)
'''

'''
documents = get_document()
word_features = filter_stop_words(documents)
test_set, classifier = NOT_features_train(documents, word_features, negationwords)
analyze_measures(test_set, classifier)
'''


SLpath = 'subjclueslen1-HLTEMNLP05.tff'
documents = get_document()
word_features = filter_stop_words(documents)
SL = readSubjectivity(SLpath)
test_set, classifier = new_SL_features_train(documents, word_features, SL)
classifier.show_most_informative_features(30)
analyze_measures(test_set, classifier)


pos_sentences = []
neg_sentences = []
csv_data = pd.read_csv('13501-27721 upload_revised.csv')  # use panda to read the csv file
for num in range(0, 14220):
    review_text_data = csv_data.loc[num, 'review_text']
    business_name_data = csv_data.loc[num, 'business_name']
    sentences = split_sentences(review_text_data)  # split text into sentences
    for sentence in sentences:
        filter_words = preprocess_data(sentence)  # preprocess the data
        inputfeatureset = document_features_contain(filter_words, word_features)  # get review data features
        if(classifier.classify(inputfeatureset) == 'pos'):
            pos_sentences.append(sentence)
        else:
            neg_sentences.append(sentence)
print(pos_sentences)
print(neg_sentences)

file_pos = open('pos_sentences.txt', mode='w')  # store positive sentences
file_neg = open('neg_sentences.txt', mode='w')  # store negative sentences
for pos_sentence in pos_sentences:
    pos_sentence = pos_sentence.replace("\r\n", " ")
    pos_sentence = pos_sentence.replace("\r", " ")
    pos_sentence = pos_sentence.replace("\n", " ")
    if(pos_sentence == " "):
        continue
    file_pos.write(pos_sentence)
    file_pos.write('\n')
file_pos.close()
for neg_sentence in neg_sentences:
    neg_sentence = neg_sentence.replace("\r\n", " ")
    neg_sentence = neg_sentence.replace("\r", " ")
    neg_sentence = neg_sentence.replace("\n", " ")
    if(neg_sentence == " "):
        continue
    file_neg.write(neg_sentence)
    file_neg.write('\n')
file_neg.close()
