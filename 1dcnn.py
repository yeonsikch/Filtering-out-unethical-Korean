from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import jamotools #자모 단위 토큰화
import re
import gluonnlp as nlp
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


class CharDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, mode):
        self.sentences = [line[sent_idx] for line in dataset]
        self.labels = [int(line[label_idx]) for line in dataset]
        self.korean = re.compile('[^1!ㄱ-ㅣ가-힣]+')
        self.mode = mode
        self.vocab = self.make_vocab()
        self.vocab_size = len(self.vocab)
        self.q3 = self.get_q3()
        self.char2idx = {u:i for i, u in enumerate(self.vocab)}
        self.idx2char = {i:u for i, u in enumerate(self.vocab)}
        self.max_len = self.find_max_len()
        
    def __getitem__(self, i):
        return (self.preprocess_sentence(self.sentences[i]), torch.tensor(self.labels[i]).to(torch.float32))
    
    def __len__(self):
        return len(self.labels)
    
    def make_vocab(self):
        vocab = ''
        for sentence in self.sentences:
            vocab+=sentence
        vocab = self.make_token(vocab)
        vocab = set(vocab)
        vocab = sorted(vocab)
        vocab.append('<UNK>') #######
        vocab.append('<PAD>')
        return vocab
    
    def make_token(self, sentence):
        if self.mode == 'jamo':
            chars = self.korean.sub('', jamotools.split_syllables(sentence))
            return list(chars)
        elif self.mode == 'char':
            chars = self.korean.sub('', sentence)
            return list(chars)
    
    def preprocess_sentence(self, sentence):
        chars = self.make_token(sentence)
        if len(chars) < self.q3:
            need_pad = self.q3 - len(chars)
            chars.extend(['<PAD>']*need_pad)
        else:
            chars = chars[:self.q3]
        chars = torch.tensor([self.char2idx[x] for x in chars]).to(torch.int64)
        return chars
    
    def find_max_len(self):
        return max(len(self.make_token(item)) for item in self.sentences)
    
    def find_max_idx(self):
        return self.sentences[np.argmax([len(self.make_token(item)) for item in self.sentences])]

    
    def get_q3(self):
        values = np.array([len(self.make_token(x)) for x in self.sentences])
        return int(np.quantile(values, 0.75))
    
    
    def plot_len(self):
        values = np.array([len(self.make_token(x)) for x in self.sentences])
        plt.hist(values, density=True, bins=80)
        plt.ylabel('count')
        plt.xlabel('length of sequence')
        plt.show()
        print('문장 최대 길이 :',self.max_len)
        results = stats.describe(values)
        print('min={}, max={}, mean={}, Q2={} Q3={}'.format(results[1][0], results[1][1], results[2],
                                                          np.median(values), np.quantile(values, 0.75)))
        
class Net(nn.Module):
    def __init__(self, vocab_size, seq_len):
        super().__init__()
        self.filters=[4,5,6]
        self.dropout_prob = 0.2
        self.embedding_dim = 500
        self.num_of_kernel = 128
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.bn1 = nn.BatchNorm1d(self.num_of_kernel)
        for i in range(len(self.filters)):
            conv = nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.num_of_kernel, kernel_size=self.filters[i])
            setattr(self, f'conv_{i}', conv)
        
        self.bn2 = nn.BatchNorm1d(self.num_of_kernel*len(self.filters))
        self.fc = nn.Linear(self.num_of_kernel*len(self.filters), 1)
        self.sigmoid = nn.Sigmoid()
        
    def get_conv(self, i):
        return getattr(self, f'conv_{i}')
    
    def forward(self, inp):
        x = self.embedding(inp)
        x = x.permute(0, 2, 1) ### embedding 을 transpose해줘야함.안하면 1d-conv이 seq 방향이 아닌, 임베딩 방향으로 진행됨.
        #x = self.bn1(x)
#         conv_results = [
#             F.relu(self.bn1(self.get_conv(i)(x))).permute(0,2,1).max(1)[0]
        conv_results = [
            F.relu(self.get_conv(i)(x)).permute(0,2,1).max(1)[0]
        for i in range(len(self.filters))]
        x = torch.cat(conv_results, 1)
        x = F.dropout(x, p=self.dropout_prob)    
        #x = self.bn2(x)
        x = self.fc(x)
        print(x)
        x = self.sigmoid(x)
        print(x)
        x = x.squeeze()
        
        return x
		
learning_rate = 4e-4
epochs = 6
batch_size = 64
prefix = './'
mode = 'char' # jamo : 자음,모음 단위로 토큰화. char : 한글자 단위로 토큰화

device = torch.device('cuda')

dataset_train = nlp.data.TSVDataset(prefix+'train_hate_dataset_v2.txt')
dataset_test = nlp.data.TSVDataset(prefix+'test_hate_dataset_v2.txt')

data_train = CharDataset(dataset_train, 0, 1, mode=mode)
data_test = CharDataset(dataset_test, 0, 1, mode=mode)

train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, drop_last=True)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, drop_last=True)

model = Net(data_train.vocab_size, data_train.get_q3())
model.to(device)
criterion = nn.BCELoss()
#optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
#criterion = nn.CrossEntropyLoss()
#optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
#optimizer = optim.SGD(model.parameters(), lr=learning_rate)
optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, momentum=0.9)

for epoch in range(epochs):
    
    running_loss = 0.0
    correct = 0
    y_true, y_pred = [], []
    model.train()
    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        #print('losses :', loss)
        running_loss = loss.item()
        loss.backward()
        optimizer.step()
        pred = (outputs>0.5).to(torch.float)
        y_pred.extend(pred)
        y_true.extend(labels)
        if i % 50 == 49:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 50))
            running_loss = 0.0
    correct = sum([(x==y) for x,y in zip(y_pred, y_true)])
    print("epoch {} train acc {}".format(epoch+1, correct / (len(y_pred))))
            
    model.eval()
    y_true, y_pred = [], []
    running_loss = 0.0
    correct = 0
    for i, data in enumerate(test_dataloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        pred = (outputs>0.5).to(torch.float)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        y_pred.extend(pred)
        y_true.extend(labels)
        if i % 50 == 49:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 50))
            running_loss = 0.0
    correct = sum([(x==y) for x,y in zip(y_pred, y_true)])
    print("\tepoch {} test acc {}".format(epoch+1, correct / (len(y_pred))))
    
print('Finished Training')