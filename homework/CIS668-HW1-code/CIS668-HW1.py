import nltk
from nltk import FreqDist
from nltk.corpus import PlaintextCorpusReader
import re
from nltk.collocations import *
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


def alpha_filter(w):
    # pattern to match a word of non-alphabetical characters
    pattern = re.compile('^[^a-z]+$')
    if pattern.match(w):
        return True
    else:
        return False


def get_part_content(filename):
    # get text file content and return the string
    corpus = PlaintextCorpusReader('.', '.*\.txt')
    content = corpus.raw(filename)
    return content


def get_word_pos(word_pos_tag):
    # get the words pos tags from pos_tag and return the corresponding wordnet tags
    if word_pos_tag.startswith('J'):
        return wordnet.ADJ
    elif word_pos_tag.startswith('V'):
        return wordnet.VERB
    elif word_pos_tag.startswith('N'):
        return wordnet.NOUN
    elif word_pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def do_word_tokenize(content):
    # do word tokenizing process
    tokens = nltk.word_tokenize(content)
    return tokens


def do_lower(content):
    # set all words as lowercase
    words = [w.lower() for w in content]
    return words


def get_alphabetical_words(content):
    # get alphabetical words
    alpha_words = [w for w in content if not alpha_filter(w)]
    return alpha_words


def get_stopwords():
    # get stopwords
    stop_words = nltk.corpus.stopwords.words('english')
    stop = open('./Smart.English.stop', 'r')
    stop_text = stop.read()
    stop.close()
    c_stopwords = nltk.word_tokenize(stop_text)
    c_stopwords.extend(["'m", "n't", "'s", "make"])
    stop_words.extend(c_stopwords)
    stop_words = list(set(stop_words))
    return stop_words


def filter_content(content, filter_words):
    # filter special words in content
    filtered_words = [w for w in content if w not in filter_words]
    return filtered_words


def do_print_top_50(content):
    # print top 50 frequency
    freq_dist = FreqDist(content)
    top_keys = freq_dist.most_common(50)
    for pair in top_keys:
        print(pair)
    print('-----------------------------------')


def get_bi_gram_association_measures():
    # get bigram measures
    measures = nltk.collocations.BigramAssocMeasures()
    return measures


def finder_filter(content):
    # apply filter to finder
    finder = BigramCollocationFinder.from_words(content)
    finder.apply_freq_filter(5)
    return finder


def print_bigram_score(bigram_measures, finder):
    # print scores which are sorted into order by decreasing frequency
    scored = finder.score_ngrams(bigram_measures.raw_freq)
    for one_score in scored[:50]:
        print(one_score)
    print('-----------------------------------')


def print_PMI_measures(bigram_measures, finder):
    # print PMI measures scores
    scored = finder.score_ngrams(bigram_measures.pmi)
    for one_score in scored[:50]:
        print(one_score)


# get content in state_union_part text file
onePartContent = get_part_content('state_union_part1.txt')
# onePartContent = get_part_content('state_union_part2.txt')

# do word tokenize of the raw content
tokens_content = do_word_tokenize(onePartContent)

# get the pos tags for all words
words_tags = pos_tag(tokens_content)

# do lemmatization
word_net_le = WordNetLemmatizer()
lemmatization_words = []
for tag in words_tags:
    word_pos = get_word_pos(tag[1]) or wordnet.NOUN
    lemmatization_words.append(word_net_le.lemmatize(tag[0], pos=word_pos))

# set all the words in the content to be as lowercase
lower_words = do_lower(lemmatization_words)

# get alphabetical words
alphabetical_words = get_alphabetical_words(lower_words)

# get stop words
stopwords = get_stopwords()

# filter the stop words
filter_stopwords = filter_content(alphabetical_words, stopwords)

# print out the top 50 words
do_print_top_50(filter_stopwords)

# get bigram measures
bi_measures = get_bi_gram_association_measures()

# get the finder
finder = BigramCollocationFinder.from_words(filter_stopwords)

# print out the top 50 bigram score
print_bigram_score(bi_measures, finder)

# apply filter to finder
finder1 = finder_filter(filter_stopwords)

# print out the top 50 PMI measures score
print_PMI_measures(bi_measures, finder1)
