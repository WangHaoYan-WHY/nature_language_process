import nltk
import re
from nltk.corpus import treebank

my_grammar = nltk.CFG.fromstring("""
  S -> WRB SQ | NP VP | VP | NP ADVP VP
  WRB -> "Why"
  JJ -> "last"
  NN -> "month"
  RB -> "again" | "not" | "always"
  UH -> "Please"
  PRP -> "their"
  INTJ -> UH
  SQ -> V NP VP
  VP -> V NP | V PP NP ADVP | INTJ VP2 ADVP
  ADVP -> RB
  PP -> To NP
  V -> "do" | "have" | "went" | "chase"
  NP -> Det | Prop | NNP CC NNP | NNP | JJ NN | N | PRP N
  VP2 -> V RB VP3
  VP3 -> V NP
  Prop -> "you"
  NNP -> "Bob" | "Mary" | "France"
  Det -> "this" | "that"
  N -> "Dogs" | "tails"
  To -> "to"
  CC -> "and"
  """)
##  four sentences and three other different sentences
rd_parser = nltk.RecursiveDescentParser(my_grammar)
sentlist1 = 'Why do you have this'.split()
print('Followings are the four sentences in the Homework:')
print('Why do you have this')
for tree in rd_parser.parse(sentlist1):
	print (tree)
print('--------------------------------')
sentlist2 = 'Bob and Mary went to France last month again'.split()
print('Bob and Mary went to France last month again')
for tree in rd_parser.parse(sentlist2):
	print (tree)
print('--------------------------------')
sentlist3 = 'Please do not do that again'.split()
print('Please do not do that again')
for tree in rd_parser.parse(sentlist3):
	print (tree)
print('--------------------------------')
sentlist4 = 'Dogs always chase their tails'.split()
print('Dogs always chase their tails')
for tree in rd_parser.parse(sentlist4):
	print (tree)
print('--------------------------------')
print('Followings are the three other sentences:')
sentlist5 = 'Dogs went to France last month again'.split()
print('FDogs went to France last month again')
for tree in rd_parser.parse(sentlist5):
	print (tree)
print('--------------------------------')
sentlist6 = 'Bob and Mary always chase their tails'.split()
print('Bob and Mary always chase their tails')
for tree in rd_parser.parse(sentlist6):
	print (tree)
print('--------------------------------')
sentlist7 = 'France always chase their tails'.split()
print('France always chase their tails')
for tree in rd_parser.parse(sentlist7):
	print (tree)
print('--------------------------------')
from nltk import induce_pcfg
from nltk import Nonterminal
S = Nonterminal('root')
productions = []
for tree in rd_parser.parse(sentlist1):
    productions += tree.productions()
for tree in rd_parser.parse(sentlist2):
    productions += tree.productions()
for tree in rd_parser.parse(sentlist3):
    productions += tree.productions()
for tree in rd_parser.parse(sentlist4):
    productions += tree.productions()
grammar = induce_pcfg(S, productions)
## print(grammar)

prob_grammar = nltk.PCFG.fromstring("""
  S -> WRB SQ[0.25] | NP VP[0.25]| VP[0.25] | NP ADVP VP[0.25]
  WRB -> "Why"[1.0]
  JJ -> "last"[1.0]
  NN -> "month"[1.0]
  RB -> "again"[0.5] | "not"[0.25] | "always"[0.25]
  UH -> "Please"[1.0]
  PRP -> "their"[1.0]
  INTJ -> UH[1.0]
  SQ -> V NP VP[1.0]
  VP -> V NP[0.5] | V PP NP ADVP[0.25] | INTJ VP2 ADVP[0.25]
  ADVP -> RB[1.0]
  PP -> To NP[1.0]
  V -> "do"[0.5] | "have"[0.166667] | "went"[0.166667] | "chase"[0.166666]
  NP -> Det[0.25] | Prop[0.125] | NNP CC NNP[0.125] | NNP[0.125] | JJ NN[0.125] | N[0.125] | PRP N[0.125]
  VP2 -> V RB VP3[1.0]
  VP3 -> V NP[1.0]
  Prop -> "you"[1.0]
  NNP -> "Bob"[0.333333] | "Mary"[0.333333] | "France"[0.333334]
  Det -> "this"[0.5] | "that"[0.5]
  N -> "Dogs"[0.5] | "tails"[0.5]
  To -> "to"[1.0]
  CC -> "and"[1.0]
  """)


def do_word_tokenize(content):
    # do word tokenizing process
    tokens = nltk.word_tokenize(content)
    return tokens

print('Followings are the probabilistic context-free grammar results:')
## probabilistics context grammer
viterbi_parser = nltk.ViterbiParser(prob_grammar)
content1 = "Why do you have this"
# do word tokenize of the raw content
tokens_content = do_word_tokenize(content1)
print('Why do you have this')
for tree in viterbi_parser.parse(tokens_content):
	print(tree)
print('--------------------------------')

content2 = "Bob and Mary went to France last month again"
# do word tokenize of the raw content
tokens_content = do_word_tokenize(content2)
print('Bob and Mary went to France last month again')
for tree in viterbi_parser.parse(tokens_content):
	print(tree)
print('--------------------------------')

content3 = "Please do not do that again"
# do word tokenize of the raw content
tokens_content = do_word_tokenize(content3)
print('Please do not do that again')
for tree in viterbi_parser.parse(tokens_content):
	print(tree)
print('--------------------------------')

content4 = "Dogs always chase their tails"
# do word tokenize of the raw content
tokens_content = do_word_tokenize(content4)
print('Dogs always chase their tails')
for tree in viterbi_parser.parse(tokens_content):
	print(tree)
print('--------------------------------')