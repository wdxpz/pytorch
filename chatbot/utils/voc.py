from chatbot.config import PAD_token, SOS_token, EOS_token


class VOC(object):
    '''
    vocabulary of all the words, provide:
    word2index,
    index2word,
    trim
    '''

    def __init__(self, name):
        super(VOC, self).__init__()
        self.name = name
        self.trimmed = False
        self._initVoc()

    def _initVoc(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.index2word[self.num_words] = word
            self.word2count[word] = 1
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def trim(self, min_account):
        if self.trimmed:
            return

        self.trimmed = True
        keep_words = []

        for word in self.word2count:
            if self.word2count[word] >= min_account:
                keep_words.append(word)

        self._initVoc()
        for word in keep_words:
            self.addWord(word)





