import codecs
import csv
import itertools
import os
import re
import unicodedata

import numpy as np
import torch

from chatbot.config import MOVIE_LINES_FIELDS, MOVIE_CONVERSATIONS_FIELDS, CORPUS_DIR, MIN_COUNT, MAX_LENGTH, \
    CORPUS_NAME, EOS_token, PAD_token
from chatbot.utils.voc import VOC


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


def filterPairs(pairs, max_length=MAX_LENGTH):
    return [pair for pair in pairs if
            (len(pair[0].split(' ')) < max_length and len(pair[1].split(' ')) < max_length)]


def indexFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]


class DataLoader(object):

    @staticmethod
    def formatMovieData(saveFilename='formatted_movie_lines.txt'):

        def loadLines(filename, fields=MOVIE_LINES_FIELDS):
            lines = {}
            with open(filename, 'r', encoding='iso-8859-1') as f:
                for line in f:
                    values = line.split(' +++$+++ ')
                    lineObj = {}
                    for i, field in enumerate(fields):
                        lineObj[field] = values[i]
                    lines[lineObj['lineID']] = lineObj

            return lines

        def loadConversations(filename, lines, fields=MOVIE_CONVERSATIONS_FIELDS):
            conversations = []
            with open(filename, 'r', encoding='iso-8859-1') as f:
                for line in f:
                    values = line.split(' +++$+++ ')
                    convObj = {}
                    for i, field in enumerate(fields):
                        convObj[field] = values[i]

                    utterance_id_pattern = re.compile('L[0-9]+')
                    lineIDs = utterance_id_pattern.findall(convObj['utteranceIDs'])
                    convObj['lines'] = []
                    for lineID in lineIDs:
                        convObj['lines'].append(lines[lineID])

                    conversations.append(convObj)

            return conversations

        def extractSentencePairs(conversations):
            qa_pairs = []
            for conversation in conversations:
                for i in range(len(conversation['lines'])-1):
                    inputLine = conversation['lines'][i]['text'].strip()
                    targetLine = conversation['lines'][i+1]['text'].strip()
                    if inputLine and targetLine:
                        qa_pairs.append([inputLine, targetLine])

            return qa_pairs

        saveFile = os.path.join(CORPUS_DIR, saveFilename)

        delimiter = '\t'
        #Unescape the delimiter
        delimiter = str(codecs.decode(delimiter, 'unicode_escape'))

        print('\nProcessing corputs...')
        lines = loadLines(os.path.join(CORPUS_DIR, 'movie_lines.txt'))
        print('\nLoading conversatons...')
        conversations = loadConversations(os.path.join(CORPUS_DIR, 'movie_conversations.txt'), lines)
        print('\nBuilding sentence pairs')
        sentence_pairs = extractSentencePairs(conversations)
        print('Read {} sentence pairs'.format(len(sentence_pairs)))

        #filter sentence pairs
        sentence_pairs = filterPairs([[normalizeString(s) for s in pair] for pair in sentence_pairs])
        print("Trimmed to {} sentence pairs".format(len(sentence_pairs)))

        #write filtered conversatoins pairs to csv file
        print('\nWriting conversation pairs to formatted csv file...')
        with open(saveFile, 'w', encoding='utf-8') as outputfile:
            writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
            for pair in sentence_pairs:
                writer.writerow(pair)

        return sentence_pairs

    @staticmethod
    def loadMovieData(formattedFilename='formatted_movie_lines.txt', min_count=MIN_COUNT, max_length=MAX_LENGTH):

        def trimRareWords(voc, pairs):
            voc.trim(min_count)

            keep_pairs = []
            for pair in pairs:
                input, target = pair[0], pair[1]
                keep_input = True
                keep_target = True

                for word in input.split(' '):
                    if word not in voc.word2index:
                        keep_input = False
                        break
                if keep_input:
                    for word in target.split(' '):
                        if word not in voc.word2index:
                            keep_target = False
                            break
                if keep_input and keep_target:
                    keep_pairs.append(pair)

            print('Trimmed from {} pairs to {}, {:.4f} of total'.format(
                len(pairs), len(keep_pairs), len(keep_pairs)/len(pair)))

            return voc, keep_pairs


        formattedFile = os.path.join(CORPUS_DIR, formattedFilename)
        print('Start loading training data ...')
        if not os.path.exists(formattedFile):
            pairs = DataLoader.formatMovieData(saveFilename=formattedFilename)
        else:
            lines = open(formattedFile, encoding='utf-8').read().strip().split('\n')
            pairs = [l.split('\t') for l in lines]
            print('Read {} filtered sentence pairs'.format(len(pairs)))

        print('Building vocabulary...')
        voc = VOC(CORPUS_NAME)
        for pair in pairs:
            voc.addSentence(pair[0])
            voc.addSentence(pair[1])
        print('Counted words: ', voc.num_words)

        voc, pairs = trimRareWords(voc, pairs)

        return voc, pairs

    @staticmethod
    def batch2TrainData(voc, pair_batch):

        def zeroPadding(batch):
            #batch is a list vary-length sequences with one max length sentence: max_length, list size: batch_size
            #after padding the shape will be [max_length, batch_size]
            return list(itertools.zip_longest(*batch, fillvalue=PAD_token))

        def binaryMask(padded):
            padded_list = np.asarray(padded)
            binary_mask = np.where(padded_list!=PAD_token, 1, 0).tolist()
            return binary_mask

        def inputVar(input_batch, voc):
            indexes_batch = [indexFromSentence(voc, sentence) for sentence in input_batch]
            lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
            padded_list = zeroPadding(indexes_batch)
            padded = torch.LongTensor(padded_list)
            return padded, lengths

        def outputVar(output_batch, voc):
            indexes_batch = [indexFromSentence(voc, sentence) for sentence in output_batch]
            max_len = max([len(indexes) for indexes in indexes_batch])
            padded_list = zeroPadding(indexes_batch)
            mask = binaryMask(padded_list)
            mask = torch.BoolTensor(mask)
            indexes_batch = torch.LongTensor(padded_list)
            return indexes_batch, mask, max_len


        #sort pair_batch for pack_padded_sequence requirement
        pair_batch.sort(key=lambda x: len(x[0].split(' ')), reverse=True)
        input_batch, output_batch = [], []
        for pair in pair_batch:
            input_batch.append(pair[0])
            output_batch.append(pair[1])

        inp, lengths = inputVar(input_batch, voc)
        output, mask, max_target_len = outputVar(output_batch, voc)

        return inp, lengths, output, mask, max_target_len
