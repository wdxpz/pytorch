import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
import torch.nn.functional as F

from TextClassify.config import TextClassifier_Config, DATA_DIR
from TextClassify.network import Network


def convert_batch(batch):
    labels = torch.tensor([entry[0] for entry in batch])

    texts = [entry[1] for entry in batch]

    offsets = [0] + [len(entry) for entry in texts]
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)

    if offsets[0] != 0:
        raise Exception('offsets[0]!=0')

    texts = torch.cat(texts)

    return texts, offsets, labels


class TextClassifier(object):

    def __init__(self, config=TextClassifier_Config):
        super(TextClassifier, self).__init__()

        self.works = config['workers']
        self.ngrams = config['ngrams']
        self.batch_size = config['batch_size']
        self.num_epochs = config['num_epochs']
        self.train_valid_ration = config['train_valid_ratio']
        self.lr = config['lr']
        self.embed_size = config['embed_size']
        self.train_dataset, self.test_dataset, self.max_sequence_len = self._init_data()
        self.vocab_size = len(self.train_dataset.get_vocab())
        self.label_size = len(self.train_dataset.get_labels())

        self.ngpu = config['ngpu']
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.net = self._init_network()

    def _init_network(self):
        net = Network(self.vocab_size, self.embed_size, self.label_size)
        net.to(self.device)

        if self.device.type == 'cuda' and self.ngpu > 1:
            net = nn.DataParallel(net, list(range(self.ngpu)))

        return net

    def _init_data(self):

        train_set, test_set = torchtext.datasets.text_classification.DATASETS['AG_NEWS'](
            root=DATA_DIR, ngrams=self.ngrams, vocab=None)

        maxlen = 0
        for seq in train_set:
            if len(seq[1])>maxlen:
                maxlen = len(seq[1])

        return train_set,  test_set, maxlen

    def _convert_batch(self, batch):
        labels = torch.tensor([entry[0] for entry in batch])

        texts = [entry[1] for entry in batch]

        offsets = [0] + [len(entry) for entry in texts]
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)

        texts = torch.cat(texts)

        return texts, offsets, labels

    def _convert_batch2(self, batch):
        labels = torch.tensor([entry[0] for entry in batch])

        texts = [F.pad(entry[1], (0, self.max_sequence_len-len(entry[1]))) for entry in batch]

        texts = nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0)

        return texts, labels


    def train(self):
        criterion = nn.CrossEntropyLoss().to(self.device)
        optimizer = optim.SGD(self.net.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

        train_size = int(len(self.train_dataset)*self.train_valid_ration)
        valid_size = len(self.train_dataset) - train_size
        train_set, valid_set = torch.utils.data.random_split(self.train_dataset, [train_size,valid_size])
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True,
                                                   num_workers=self.works, collate_fn=self._convert_batch2)
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=self.batch_size, shuffle=True,
                                                 num_workers=self.works, collate_fn=self._convert_batch2)



        train_accu = 0
        train_loss = 0

        print_step = 1000

        for epoch in range(1, self.num_epochs+1):

            tsampels = 0
            for i, batch in enumerate(train_loader):


                loss, accu, nsamples = self._train_func(batch, criterion, optimizer)
                train_loss += loss
                train_accu += accu
                tsampels += nsamples

                if (i+1) % print_step == 0:
                    valid_loss, valid_accu, vsamples = self._valid_func(valid_loader, criterion)
                    print('epoch:[{}/{}] step:[{}/{}]\ttrain_loss: {:.4f}\ttrain_accu: {:.4f}\t'
                          'valid_loss: {:.4f}\tvalid_accu: {:.4f}'.format(
                        epoch, self.num_epochs, i+1, len(train_loader),
                        train_loss/tsampels, train_accu/tsampels, valid_loss/vsamples, valid_accu/vsamples))
                    train_loss = 0
                    train_accu = 0
                    tsampels = 0

        scheduler.step()


    def _train_func(self, batch, criterion, optimizer):
        optimizer.zero_grad()

        text, labels = batch[0], batch[1],
        text, labels = text.to(self.device), labels.to(self.device)

        logits = self.net(text, offsets=None)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        if loss > 100:
            print('huge loss')
        loss = loss.item()
        accu = (logits.argmax(dim=1) == labels).sum().item()
        nsamples = len(labels)

        return loss, accu, nsamples


    def _valid_func(self, dataloader, criterion):
        nsampels = 0
        loss = 0.0
        accu = 0.0
        with torch.no_grad():
            for i, (text, labels) in enumerate(dataloader):
                text, labels = text.to(self.device), labels.to(self.device)
                logits = self.net(text, offsets=None)
                loss += criterion(logits, labels).item()
                accu += (logits.argmax(dim=1) == labels).sum().item()
                nsampels += len(labels)

        return loss, accu, nsampels





