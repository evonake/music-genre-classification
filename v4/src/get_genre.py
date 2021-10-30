import os
import numpy as np
import torch
import sys

from collections import Counter
from sklearn.preprocessing import LabelEncoder

from librosa.core import load
from librosa.feature import melspectrogram
from librosa import power_to_db

from model import genreNet
from config import GENRES, DATAPATH, MODELPATH
from config import GENRES
from data import Data
from set import Set

from torch.autograd import Variable

import warnings
warnings.filterwarnings('ignore')

def main(argv):

    if len(argv) != 1:
        print('Usage: python3 get_genre.py audiopath')
        exit()

    genrefile_name = argv[0][58:]

    le = LabelEncoder().fit(GENRES)
    # ------------------------------- #
    ## LOAD TRAINED GENRENET MODEL
    net         = genreNet()
    net.load_state_dict(torch.load(MODELPATH, map_location='cpu'))
    # ------------------------------- #
    ## LOAD AUDIO
    audio_path  = argv[0]
    y, sr       = load(audio_path, mono=True, sr=22050)
    # ------------------------------- #
    ## GET CHUNKS OF AUDIO SPECTROGRAMS
    S           = melspectrogram(y, sr).T
    S           = S[:-1 * (S.shape[0] % 128)]
    num_chunk   = S.shape[0] / 128
    data_chunks = np.split(S, num_chunk)
    # ------------------------------- #
    ## CLASSIFY SPECTROGRAMS
    genres = list()
    for i, data in enumerate(data_chunks):
        data    = torch.FloatTensor(data).view(1, 1, 128, 128)
        preds   = net(data)
        pred_val, pred_index    = preds.max(1)
        pred_index              = pred_index.data.numpy()
        pred_val                = np.exp(pred_val.data.numpy()[0])
        pred_genre              = le.inverse_transform(pred_index).item()
        if pred_val >= 0.5:
            genres.append(pred_genre)
    # ------------------------------- #
    s           = float(sum([v for k,v in dict(Counter(genres)).items()]))
    pos_genre   = sorted([(k, v/s*100 ) for k,v in dict(Counter(genres)).items()], key=lambda x:x[1], reverse=True)
    for i in range(len(pos_genre)):
        if pos_genre[i][0] == genrefile_name[:len(pos_genre[i][0])] and round(pos_genre[i][1]) == round(pos_genre[0][1]):
            return 1
        if round(pos_genre[i][1]) != round(pos_genre[0][1]):
            return 0
    return 0
    # for genre, pos in pos_genre:
    #     print('%10s: \t%.2f\t%%' % (genre, pos))
    # return


def test_acc():
    data    = Data(GENRES, DATAPATH)
    data.load()

    set_ = Set(data)
    set_.load()

    x_train, y_train    = set_.get_train_set()
    x_valid, y_valid    = set_.get_valid_set()
    x_test,  y_test     = set_.get_test_set()
    TRAIN_SIZE, VALID_SIZE, TEST_SIZE = len(x_train), len(x_valid), len(x_test)
    BATCH_SIZE  = 16

    net         = genreNet()
    net.load_state_dict(torch.load(MODELPATH, map_location='cpu'))

    # dictionary key = actual answer, value = guessed value sorted by index
    actualvguessed = {0: [0] * 10,
                      1: [0] * 10,
                      2: [0] * 10,
                      3: [0] * 10,
                      4: [0] * 10,
                      5: [0] * 10,
                      6: [0] * 10,
                      7: [0] * 10,
                      8: [0] * 10,
                      9: [0] * 10,
    }

    net.eval()
    inp_train, out_train    = Variable(torch.from_numpy(x_train)).float(), Variable(torch.from_numpy(y_train)).long()
    inp_valid, out_valid    = Variable(torch.from_numpy(x_valid)).float(), Variable(torch.from_numpy(y_valid)).long()
    inp_test, out_test      = Variable(torch.from_numpy(x_test)).float(),  Variable(torch.from_numpy(y_test)).long()
    train_sum, valid_sum, test_sum = 0, 0, 0
    for i in range(0, TRAIN_SIZE, BATCH_SIZE):
        pred_train       = net(inp_train[i:i + BATCH_SIZE])
        indices_train    = pred_train.max(1)[1]
        train_sum        += (indices_train == out_train[i:i + BATCH_SIZE]).sum().data.cpu().numpy()

    for i in range(0, VALID_SIZE, BATCH_SIZE):
        pred_valid       = net(inp_valid[i:i + BATCH_SIZE])
        indices_valid    = pred_valid.max(1)[1]
        valid_sum        += (indices_valid == out_valid[i:i + BATCH_SIZE]).sum().data.cpu().numpy()

    for i in range(0, TEST_SIZE, BATCH_SIZE):
        pred_test        = net(inp_test[i:i + BATCH_SIZE])
        indices_test     = pred_test.max(1)[1]
        test_sum         += (indices_test == out_test[i:i + BATCH_SIZE]).sum().data.cpu().numpy()

        for j in range(len(indices_test)):
            if not indices_test[j] == out_test[i]:
                actualvguessed[out_test[i].item()][indices_test[j].item()] += 1

    train_accuracy  = train_sum / float(TRAIN_SIZE) * 100
    valid_accuracy  = valid_sum / float(VALID_SIZE) * 100
    test_accuracy   = test_sum  / float(TEST_SIZE)  * 100
    print('train_acc: %.2f (%d / %d) ; valid_acc: %.2f (%d / %d) ; test_acc: %.2f (%d / %d)' % \
            (train_accuracy, train_sum, TRAIN_SIZE, valid_accuracy, valid_sum, VALID_SIZE, test_accuracy, test_sum, TEST_SIZE))
    return actualvguessed


if __name__ == '__main__':
    # main(sys.argv[1:])

    data = test_acc()
    data['total'] = [0] * 10
    for i in range(10):
        for j in range(10):
            data['total'][j] += data[i][j]

    genres = {
        0: 'blues    ',
        1: 'classical',
        2: 'country  ',
        3: 'disco    ',
        4: 'hiphop   ',
        5: 'jazz     ',
        6: 'metal    ',
        7: 'pop      ',
        8: 'reggae   ',
        9: 'rock     '
    }

    print('actual: guessed')
    for i in range(10):
        print(genres[i], ': ', end='')
        for j in range(10):
            print(str(data[i][j]) if len(str(data[i][j])) == 3 else ' ' * (3 - len(str(data[i][j]))) + str(data[i][j]), end=' ')
        print()
    print('total    ', ': ', end='')
    for j in range(10):
        print(str(data['total'][j]) if len(str(data['total'][j])) == 3 else ' ' * (3 - len(str(data['total'][j]))) + str(data['total'][j]), end=' ')
    print()
