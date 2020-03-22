import json
import numpy as np
from collections import Counter, defaultdict

with open('./training_corpus.json','r') as infile:
    o = json.load(infile)
    percentage = .98
    o = o[:round(len(o)/2)]
    with open('training_corpus1.json', 'w') as outfile:
        json.dump(o[:round(len(o)*percentage)], outfile)
    with open('testing_corpus.json', 'w') as outfile:
        json.dump(o[round(len(o)*percentage):len(o)], outfile)

train_path = './training_corpus1.json'
test_path = './testing_corpus.json'


class Corpus(object):
    """
    This class creates a corpus object read off a .json file consisting of a list of lists,
    where each inner list is a sentence encoded as a list of strings.
    """

    def __init__(self, path, t, n, bos_eos=True, vocab=None):

        """
        DON'T TOUCH THIS CLASS!
        IT'S HERE TO SHOW THE PROCESS, YOU DON'T NEED TO ANYTHING HERE.

        A Corpus object has the following attributes:
         - vocab: set or None (default). If a set is passed, words in the input file not
                         found in the set are replaced with the UNK string
         - path: str, the path to the .json file used to build the corpus object
         - t: int, words with frequency count < t are replaced with the UNK string
         - ngram_size: int, 2 for bigrams, 3 for trigrams, and so on.
         - bos_eos: bool, default to True. If False, bos and eos symbols are not
                     prepended and appended to sentences.
         - sentences: list of lists, containing the input sentences after lowercasing and
                         splitting at the white space
         - frequencies: Counter, mapping tokens to their frequency count in the corpus
        """

        self.vocab = vocab
        self.path = path
        self.t = t
        self.ngram_size = n
        self.bos_eos = bos_eos

        self.sentences = self.read()
        # output --> [['i', 'am', 'home' '.'], ['you', 'went', 'to', 'the', 'park', '.'], ...]

        # example for testing: {ba, ab, ba, aa, ac, cb, bc, ca, ac, ca, ac}

        """
        self.sentences = [
            ['b', 'a'], ['a', 'b'], ['b', 'a'], ['a', 'a'],
            ['a', 'c'], ['c', 'b'], ['b', 'c'], ['c', 'a'],
            ['a', 'c'], ['c', 'a'], ['a', 'c']
        ]

        """

        self.frequencies = self.freq_distr()
        # output --> Counter('the': 485099, 'of': 301877, 'i': 286549, ...)
        # the numbers are made up, they aren't the actual frequency counts

        if self.t or self.vocab:
            # input --> [['i', 'am', 'home' '.'], ['you', 'went', 'to', 'the', 'park', '.'], ...]
            self.sentences = self.filter_words()
            # output --> [['i', 'am', 'home' '.'], ['you', 'went', 'to', 'the', 'UNK', '.'], ...]
            # supposing that park wasn't frequent enough or was outside of the training
            # vocabulary, it gets replaced by the UNK string

        if self.bos_eos:
            # input --> [['i', 'am', 'home' '.'], ['you', 'went', 'to', 'the', 'park', '.'], ...]
            self.sentences = self.add_bos_eos()
            # output --> [['bos', i', 'am', 'home' '.', 'eos'],
            #             ['bos', you', 'went', 'to', 'the', 'park', '.', 'eos'], ...]

    def read(self):

        """
        Reads the sentences off the .json file, replaces quotes, lowercases strings and splits
        at the white space. Returns a list of lists.
        """

        if self.path.endswith('.json'):
            sentences = json.load(open(self.path, 'r'))
        else:
            sentences = []
            with open(self.path, 'r', encoding='latin-1') as f:
                for line in f:
                    print(line[:20])
                    # first strip away newline symbols and the like, then replace ' and " with the empty
                    # string and get rid of possible remaining trailing spaces
                    line = line.strip().translate({ord(i): None for i in '"\'\\'}).strip(' ')
                    # lowercase and split at the white space (the corpus has ben previously tokenized)
                    sentences.append(line.lower().split(' '))

        return sentences

    def freq_distr(self):

        """
        Creates a counter mapping tokens to frequency counts

        count = Counter()
        for sentence in self.sentences:
            for word in sentence:
                count[w] += 1

        """

        return Counter([word for sentence in self.sentences for word in sentence])

    def filter_words(self):

        """
        Replaces illegal tokens with the UNK string. A token is illegal if its frequency count
        is lower than the given threshold and/or if it falls outside the specified vocabulary.
        The two filters can be both active at the same time but don't have to be. To exclude the
        frequency filter, set t=0 in the class call.
        """

        filtered_sentences = []
        for sentence in self.sentences:
            filtered_sentence = []
            for word in sentence:
                if self.t and self.vocab:
                    # check that the word is frequent enough and occurs in the vocabulary
                    filtered_sentence.append(
                        word if self.frequencies[word] > self.t and word in self.vocab else 'UNK'
                    )
                else:
                    if self.t:
                        # check that the word is frequent enough
                        filtered_sentence.append(word if self.frequencies[word] > self.t else 'UNK')
                    else:
                        # check if the word occurs in the vocabulary
                        filtered_sentence.append(word if word in self.vocab else 'UNK')

            if len(filtered_sentence) > 1:
                # make sure that the sentence contains more than 1 token
                filtered_sentences.append(filtered_sentence)

        return filtered_sentences

    def add_bos_eos(self):

        """
        Adds the necessary number of BOS symbols and one EOS symbol.

        In a bigram model, you need one bos and one eos; in a trigram model you need two bos and one eos,
        and so on...
        """

        padded_sentences = []
        for sentence in self.sentences:
            padded_sentence = ['#bos#'] * (self.ngram_size - 1) + sentence + ['#eos#']
            padded_sentences.append(padded_sentence)

        return padded_sentences


class LM(object):
    """
    Creates a language model object which can be trained and tested.
    The language model has the following attributes:
     - vocab: set of strings
     - lam: float, indicating the constant to add to transition counts to smooth them (default to 1)
     - ngram_size: int, the size of the ngrams
    """

    def __init__(self, n, vocab=None, smooth='Laplace', lam=1):

        self.vocab = vocab
        self.lam = lam
        self.ngram_size = n
        if (n > 1):
            print("test")
            train_corpus_unigram = Corpus(train_path, 10, 1, bos_eos=True, vocab=None)
            self.unigram_model = LM(1, lam=self.lam)
            self.unigram_model.update_counts(train_corpus_unigram)
        if (n > 2):
            print("test")
            self.bigram_laplace_model = LM(2, lam=self.lam)
            self.bigram_laplace_model.update_counts(Corpus(train_path, 10, 2, bos_eos=True, vocab=None))

            print("test")
            self.bigram_turing_model = LM(2, lam=self.lam)
            self.bigram_turing_model.update_counts(Corpus(train_path, 10, 2, bos_eos=True, vocab=None))

    def get_ngram(self, sentence, i):

        """
        CHANGE AT OWN RISK.

        Takes in a list of string and an index, and returns the history and current
        token of the appropriate size: the current token is the one at the provided
        index, while the history consists of the n-1 previous tokens. If the ngram
        size is 1, only the current token is returned.

        Example:
        input sentence: ['bos', 'i', 'am', 'home', 'eos']
        target index: 2
        ngram size: 3

        ngram = ['bos', 'i', 'am']
        #from index 2-(3-1) = 0 to index i (the +1 is just because of how Python slices lists)

        history = ('bos', 'i')
        target = 'am'
        return (('bos', 'i'), 'am')
        """

        if self.ngram_size == 1:
            return sentence[i]
        else:
            ngram = sentence[i - (self.ngram_size - 1):i + 1]
            history = tuple(ngram[:-1])
            target = ngram[-1]
            return (history, target)

    def update_counts(self, corpus):

        """
        CHANGE AT OWN RISK.

        Creates a transition matrix with counts in the form of a default dict mapping history
        states to current states to the co-occurrence count (unless the ngram size is 1, in which
        case the transition matrix is a simple counter mapping tokens to frequencies.
        The ngram size of the corpus object has to be the same as the language model ngram size.
        The input corpus (passed by providing the corpus object) is processed by extracting ngrams
        of the chosen size and updating transition counts.

        This method creates three attributes for the language model object:
         - counts: dict, described above
         - vocab: set, containing all the tokens in the corpus
         - vocab_size: int, indicating the number of tokens in the vocabulary
        """

        if self.ngram_size != corpus.ngram_size:
            raise ValueError("The corpus was pre-processed considering an ngram size of {} while the "
                             "language model was created with an ngram size of {}. \n"
                             "Please choose the same ngram size for pre-processing the corpus and fitting "
                             "the model.".format(corpus.ngram_size, self.ngram_size))

        self.counts = defaultdict(dict) if self.ngram_size > 1 else Counter()
        for sentence in corpus.sentences:
            for idx in range(self.ngram_size - 1, len(sentence)):
                ngram = self.get_ngram(sentence, idx)
                if self.ngram_size == 1:
                    self.counts[ngram] += 1
                else:
                    # it's faster to try to do something and catch an exception than to use an if statement to check
                    # whether a condition is met beforehand. The if is checked everytime, the exception is only catched
                    # the first time, after that everything runs smoothly
                    try:
                        self.counts[ngram[0]][ngram[1]] += 1
                    except KeyError:
                        self.counts[ngram[0]][ngram[1]] = 1

        # first loop through the sentences in the corpus, than loop through each word in a sentence
        self.vocab = {word for sentence in corpus.sentences for word in sentence}
        self.vocab_size = len(self.vocab)

        if self.ngram_size == 2:
            self.calculatePropabilitiesForGoodTuring()

    def calculatePropabilitiesForGoodTuring(self):

        N = 0
        for val in self.counts.values():
            for count in val.values():
                N += count

        countdict = {}
        for val in self.counts.values():
            for count in val.values():
                if count not in countdict:
                    countdict[count] = 1
                else:
                    countdict[count] += 1

        countdict[0] = 0

        for i in self.vocab:
            if (i,) not in self.counts:
                countdict[0] += self.vocab_size
            else:
                for j in self.vocab:
                    if j not in self.counts[(i,)]:
                        countdict[0] += 1

        def estimatedProbability(r):
            if r > 15:
                return r / N
            else:
                Nr = countdict.get(r,
                                   0)  # if r exists in the dictionary, return it, else return the default value (0 in this case)
                Nr1 = countdict.get(r + 1, 0)
                return (r + 1) * (Nr1 / (N * Nr))

        estimateprobdict = {}
        for i in countdict:
            estimateprobdict[i] = estimatedProbability(i) * countdict[i]

        oldrange = 0
        for i in estimateprobdict:
            if i != 0:
                oldrange += estimateprobdict[i]

        newrange = 1 - estimateprobdict[0]

        scalar = newrange / oldrange

        newprobdict = {}
        for i in countdict:
            if i == 0:
                newprobdict[i] = estimatedProbability(i) * countdict[i]
            else:
                newprobdict[i] = estimatedProbability(i) * scalar * countdict[i]

        self.probdict = newprobdict

    def get_unigram_probability(self, ngram):

        """
        CHANGE THIS.

        Compute the probability of a given unigram in the estimated language model using
        Laplace smoothing (add k).
        """

        tot = sum(list(self.counts.values())) + (self.vocab_size * self.lam)
        try:
            ngram_count = self.counts[ngram] + self.lam
        except KeyError:
            ngram_count = self.lam
            print(ngram_count, tot)

        return ngram_count / tot

    def get_ngram_probability(self, history, target, weight = 0.8, solo=False, turing=False):

        """
        CHANGE THIS.

        Compute the conditional probability of the target token given the history, using
        Laplace smoothing (add k).

        """

        try:
            transition_count = self.counts[history][target]
        except KeyError:
            transition_count = 0

        if (turing):
            return self.probdict[transition_count]

        try:
            ngram_tot = np.sum(list(self.counts[history].values())) + (self.vocab_size * self.lam)
            try:
                transition_count = self.counts[history][target] + self.lam
            except KeyError:
                transition_count = self.lam
        except KeyError:
            transition_count = self.lam
            ngram_tot = self.vocab_size * self.lam
        if (solo):
            return transition_count / ngram_tot
        elif (n == 2):
            return (weight * transition_count / ngram_tot) \
                   + ((1 - weight) * self.unigram_model.get_unigram_probability(target))
        elif (n == 3):
            return (weight[0] * transition_count / ngram_tot) \
                   + (weight[1] * self.unigram_model.get_unigram_probability(target)) \
                   + (weight[2] * self.bigram_laplace_model.get_ngram_probability((history[1],), target, solo=True)) \
                   + (weight[3] * self.bigram_turing_model.get_ngram_probability((history[1],), target, solo=True,
                                                                          turing=True))


    def perplexity(self, test_corpus,weights):
        """
        Uses the estimated language model to process a corpus and computes the perplexity
        of the language model over the corpus.

        DON'T TOUCH THIS FUNCTION!!!
        """

        probs = []
        for sentence in test_corpus.sentences:
            for idx in range(self.ngram_size - 1, len(sentence)):
                ngram = self.get_ngram(sentence, idx)
                if self.ngram_size == 1:
                    probs.append(self.get_unigram_probability(ngram))
                else:
                    probs.append(self.get_ngram_probability(ngram[0], ngram[1], weights))

        entropy = np.log2(probs)
        # this assertion makes sure that you retrieved valid probabilities, whose log must be <= 0
        assert all(entropy <= 0)

        avg_entropy = -1 * (sum(entropy) / len(entropy))

        return pow(2.0, avg_entropy)


# example code to run a unigram model with add 0.001 smoothing. Tokens with a frequency count lower than 10
# are replaced with the UNK string
# example code to run a trigram model with add 0.001 smoothing. The same frequency threshold is applied.
n = 3

train_corpus = Corpus(train_path, 10, n, bos_eos=True, vocab=None)

trigram_model = LM(n, lam=0.001)

trigram_model.update_counts(train_corpus)

# to ensure consistency, the test corpus is filtered using the vocabulary of the trained language model
test_corpus = Corpus(test_path, None, n, bos_eos=True, vocab=trigram_model.vocab)
#weight = .8058551773478578 perplexity = 295.21
perplexity = float("inf")


for i in np.random.dirichlet(np.ones(4), size = 1000):
    tmp = trigram_model.perplexity(test_corpus, i)
    print (i, tmp)
print (i, tmp, "done")



