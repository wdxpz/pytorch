import os
import random

import torch
import torch.nn as nn

from chatbot.config import ChatBotConfig, MODEL_DIR, CORPUS_NAME, MAX_LENGTH, SOS_token
from chatbot.network import EncoderRNN, LuongAttnDecoderRNN, GreedySearchDecoder
from chatbot.utils.dataloader import DataLoader, normalizeString, indexFromSentence
from chatbot.utils.voc import VOC


class ChatBot(object):
    def __init__(self, config=ChatBotConfig):
        self.load_saved_model = config['load_saved_model']
        self.model_name = config['model_name']
        self.attn_method = config['attn_method']
        self.hidden_size = config['hidden_size']
        self.encoder_n_layers = config['encoder_n_layers']
        self.decoder_n_layers = config['decoder_n_layers']
        self.dropout = config['dropout']
        self.batch_size = config['batch_size']
        self.gradient_clip = config['gradient_clip']
        self.teacher_forcing_ratio = config['teacher_forcing_ratio']
        self.learning_rate = config['learning_rate']
        self.decoder_learning_ratio = config['decoder_learning_ratio']
        self.checkpoint_iter = config['checkpoint_iter']
        self.n_iteration = config['n_iteration']
        self.print_every = config['print_every']
        self.save_every = config['save_every']
        self.model_file = os.path.join(MODEL_DIR,
                                       '{}_{}-{}_{}_{}_checkpoint.tar'.format(self.model_name, self.encoder_n_layers,
                                                            self.decoder_n_layers, self.hidden_size, self.checkpoint_iter))
        self.start_iteration = 1
        self.voc = None
        self.train_pairs = None


        # networks
        self.checkpoint = None
        if self.load_saved_model and os.path.exists(self.model_file):
            model_loaded = True
            if torch.cuda.is_available():
                self.checkpoint = torch.load(self.model_file)
            else:
                #if loading a model trained on GPU to CPU
                self.checkpoint = torch.load(self.model_file, map_location=torch.device('cpu'))

            encoder_sd = self.checkpoint['en']
            decoder_sd = self.checkpoint['de']
            embedding_sd = self.checkpoint['embedding']
            voc_dict = self.checkpoint['voc_dict']
            self.start_iteration = self.checkpoint['iteration']+1

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.voc = VOC(CORPUS_NAME)
        self.embedding = None
        self.encoder = None
        self.decoder = None
        self.searcher = None

        #load saved module
        if self.checkpoint:
            self.voc.__dict__ = voc_dict

            self.embedding = nn.Embedding(self.voc.num_words, self.hidden_size)
            #embedding layer is not put to GPU
            self.embedding.load_state_dict(embedding_sd)

            self.encoder = EncoderRNN(self.hidden_size, self.embedding, self.encoder_n_layers, self.dropout)
            self.decoder = LuongAttnDecoderRNN(self.attn_method, self.embedding, self.hidden_size, self.voc.num_words,
                                               self.decoder_n_layers, self.dropout)
            self.encoder.load_state_dict(encoder_sd)
            self.decoder.load_state_dict(decoder_sd)

            # use appropriate device
            self.encoder = self.encoder.to(self.device)
            self.decoder = self.decoder.to(self.device)

            self.searcher = GreedySearchDecoder(self.device, self.encoder, self.decoder)

    def _init_optimizer(self):
        en_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.learning_rate)
        de_optimizer = torch.optim.Adam(self.decoder.parameters(),
                                             lr=self.learning_rate * self.decoder_learning_ratio)
        if self.checkpoint:
            en_optimizer.load_state_dict(self.checkpoint['en_opt'])
            de_optimizer.load_state_dict(self.checkpoint['de_opt'])

        # if have cuda, ocnfigure cuda to call
        for state in en_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
        for state in de_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        return en_optimizer, de_optimizer

    def _maskNLLloss(self, inp, target, mask):
        nTotals = mask.sum()
        crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1))).squeeze(1)
        loss = crossEntropy.masked_select(mask).mean()
        loss = loss.to(self.device)
        return loss, nTotals.item()

    def _train_step(self, input_variable, lengths, target_variable, mask, max_target_len,
                                   encoder_optimizer, decoder_optimizer):
        #zero gradients
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        #set device options
        input_variable = input_variable.to(self.device)
        lengths = lengths.to(self.device)
        target_variable = target_variable.to(self.device)
        mask = mask.to(self.device)

        #initialize variables
        loss = 0
        print_losses = []
        n_totals = 0

        #forward pass through encoder, encoder go through one batch at a time
        encoder_outputs, encoder_hidden = self.encoder(input_variable, lengths)

        #init variables for decoder
        ###create initial decoder input (starts with SOS tokens for each sentence)
        decoder_input = torch.LongTensor([[SOS_token for _ in range(self.batch_size)]])
        decoder_input = decoder_input.to(self.device)
        ####set initial decoder hidden state to the enocder's final hidden state
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        ###determing if using teacher forcing in this iteration
        use_teacher_forcing = True if random.random()>self.teacher_forcing_ratio else False

        #forward pass through decoder, decoder go through one step (one word of each sentence) at a time
        if use_teacher_forcing:
            for t in range(max_target_len):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                #teacher forcing: next input is current target's number t row, eg. the number t words of each sentence
                decoder_input = target_variable[t].view(1, -1)
                #calculate and accumulate loss
                mask_loss, nTotals = self._maskNLLloss(decoder_output, target_variable[t], mask[t])
                loss += mask_loss
                print_losses.append(mask_loss.item()*nTotals)
                n_totals += nTotals
        else:
            for t in range(max_target_len):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                # no teacher forcing, next input is decoder's own current output
                _, topi = decoder_output.topk(1)
                decoder_input = torch.LongTensor([[topi[i][0] for i in range(self.batch_size)]])
                decoder_input = decoder_input.to(self.device)
                # calculate and accumulate loss
                mask_loss, nTotals = self._maskNLLloss(decoder_output, target_variable[t], mask[t])
                loss += mask_loss
                print_losses.append(mask_loss.item() * nTotals)
                n_totals += nTotals

        #perform backpropatation
        loss.backward()

        #clip gradients: gradients are modified in place
        _ = nn.utils.clip_grad_norm_(self.encoder.parameters(), self.gradient_clip)
        _ = nn.utils.clip_grad_norm_(self.decoder.parameters(), self.gradient_clip)

        #adjust model weights
        encoder_optimizer.step()
        decoder_optimizer.step()

        return sum(print_losses) / n_totals

    def train(self):
        #load training data
        if self.voc is None or self.train_pairs is None:
            self.voc, self.train_pairs = DataLoader.loadMovieData()

        #create embedding, encoder, decoder if no existed model loaded
        if not self.checkpoint:
            self.embedding = nn.Embedding(self.voc.num_words, self.hidden_size)
            self.encoder = EncoderRNN(self.hidden_size, self.embedding, self.encoder_n_layers, self.dropout)
            self.decoder = LuongAttnDecoderRNN(self.attn_method, self.embedding, self.hidden_size, self.voc.num_words,
                                               self.decoder_n_layers, self.dropout)
            # use appropriate device
            self.encoder = self.encoder.to(self.device)
            self.decoder = self.decoder.to(self.device)

        #create optimizers
        encoder_optimizer, decoder_optimizer = self._init_optimizer()

        #enable dropout in train mode
        self.encoder.train()
        self.decoder.train()


        #Load batches for each iteration
        training_batches = [DataLoader.batch2TrainData(self.voc, [random.choice(self.train_pairs)
                                                                  for _ in range(self.batch_size)])
                            for _ in range(self.n_iteration)]


        print_loss = 0
        print('Traing...')
        for iteration in range(self.start_iteration, self.n_iteration+1):
            training_batch = training_batches[iteration-1]
            input_variable, lengths, target_variable, mask, max_target_len = training_batch

            loss = self._train_step(input_variable, lengths, target_variable, mask, max_target_len,
                                    encoder_optimizer, decoder_optimizer)
            print_loss += loss

            if iteration % self.print_every == 0:
                print_loss_avg = print_loss / self.print_every
                print('Iteration: {}; Percent complete: {:.1f} Avarage loss: {:.4f}'.format(
                    iteration, iteration/self.n_iteration*100, print_loss_avg))
                print_loss = 0

            if iteration % self.save_every == 0:
                model_file = os.path.join(MODEL_DIR,
                                          '{}_{}-{}_{}_{}_checkpoint.tar'.format(self.model_name, self.encoder_n_layers,
                                                                                 self.decoder_n_layers,
                                                                                 self.hidden_size,
                                                                                 iteration))
                torch.save({
                    'iteration': iteration,
                    'en': self.encoder.state_dict(),
                    'de': self.decoder.state_dict(),
                    'en_opt': encoder_optimizer.state_dict(),
                    'de_opt': decoder_optimizer.state_dict(),
                    'loss': loss,
                    'voc_dict': self.voc.__dict__,
                    'embedding': self.embedding.state_dict()

                }, model_file)

    def search(self, sentence, max_length=MAX_LENGTH):
        self.encoder.eval()
        self.decoder.eval()

        if not self.searcher:
            self.searcher = GreedySearchDecoder(self.device, self.encoder, self.decoder)

        sentence = normalizeString(sentence)
        indexes_batch = [indexFromSentence(self.voc, sentence)]
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch]).to(self.device)
        input_batch = torch.LongTensor(indexes_batch).to(self.device).transpose(0, 1)

        with torch.no_grad():
            tokens, scores = self.searcher(input_batch, lengths, max_length)

        decoded_words = [self.voc.index2word[token.item()] for token in tokens]

        output_words = [word for word in decoded_words if not (word == 'EOS' or word == 'PAD')]

        return output_words
















