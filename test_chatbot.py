import sys
from chatbot.chatbot import ChatBot


chatbot = ChatBot()
if len(sys.argv) == 1:
    chatbot.train()
else:
    sentences = sys.argv[1:]
    for sentence in sentences:
        answer = chatbot.search(sentence)
        print('\n{}\t->\t{}'.format(sentence, ' '.join(answer)))