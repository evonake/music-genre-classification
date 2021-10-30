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
warnings.filterwarnings("ignore")

def main(argv):

    if len(argv) != 1:
        print("Usage: python3 get_genre.py audiopath")
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
    #     print("%10s: \t%.2f\t%%" % (genre, pos))
    # return


def test_acc():
    data    = Data(GENRES, DATAPATH)
    data.load()

    set_ = Set(data)
    set_.load()

    x_test,  y_test     = set_.get_test_set()
    TEST_SIZE   = len(x_test)
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

    inp_test, out_test = Variable(torch.from_numpy(x_test)).float(), Variable(torch.from_numpy(y_test)).long()
    test_sum = 0
    for i in range(0, TEST_SIZE, BATCH_SIZE):
        pred_test       = net(inp_test[i:i + BATCH_SIZE])
        indices_test    = pred_test.max(1)[1]
        for j in range(len(indices_test)):
            if not indices_test[j] == out_test[i]:
                actualvguessed[out_test[i].item()][indices_test[j].item()] += 1
        test_sum        += (indices_test == out_test[i:i + BATCH_SIZE]).sum().data.cpu().numpy()
    test_accuracy   = test_sum / float(TEST_SIZE)
    print("Test acc: %.2f" % test_accuracy)
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
