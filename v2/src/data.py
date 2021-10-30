import pandas
import numpy
import pickle
import os

import librosa.display
from librosa.core import load
from librosa.feature import melspectrogram

from config import RAW_DATAPATH

import matplotlib.pyplot as plt

class Data:

    def __init__(self, genres, datapath):
        self.raw_data   = None
        self.GENRES     = genres

        self.DATAPATH   = datapath
        print('\n-> Data() object is initialized.')

    # INITIALIZES self.raw_data FROM RAW_DATAPATH AS pandas.DataFrame() OBJECT
    def make_raw_data(self):
        records = []

        # HORIZONTAL LENGTH OF INPUT (VERTICAL = 128)
        # 128 = 3 seconds
        input_length = 128

        for i, genre in enumerate(self.GENRES):
            GENREPATH = self.DATAPATH + genre + '/'
            for j, track in enumerate(os.listdir(GENREPATH)):
                TRACKPATH   = GENREPATH + track

                # if j % 10 == 9:
                #     print('%d.%s\t\t%s (%d)' % (i + 1, genre, TRACKPATH, j + 1))

                y, sr       = load(TRACKPATH, mono=True)

                # CREATE MELSPECTROGRAM OF TIME-SERIES AND STRIP TO SHAPE (128 * k, input_length)
                S           = melspectrogram(y, sr).T
                S           = S[:-1 * (S.shape[0] % input_length)].T
                S = librosa.power_to_db(S, ref=numpy.max)

                # SPLIT MELSPECTROGRAM INTO CHUNKS OF SHAPE (128, input_length) -------------------
                num_chunk   = S.shape[1] / input_length
                data_chunks = numpy.split(S, num_chunk, axis=1)

                # SHOWING SPECTROGRAM (X = TIME, Y = FREQUENCY, COLORS = AMPLITUDE)
                # fig, ax = plt.subplots()
                # img = librosa.display.specshow(data_chunks[0], x_axis='time',
                                        # y_axis='mel', sr=sr,
                                        # fmax=8000, ax=ax)
                # fig.colorbar(img, ax=ax, format='%+2.0f dB')
                # ax.set(title='Mel-frequency spectrogram')
                # plt.show()

                # return
                # ---------------------------------------------------------------------------------

                data_chunks = [(data, genre) for data in data_chunks]
                records.append(data_chunks)

        records = [data for record in records for data in record]
        self.raw_data = pandas.DataFrame.from_records(records, columns=['spectrogram', 'genre'])
        return

    # SAVES self.raw_data INTO raw_data.pkl
    def save(self):
        with open(RAW_DATAPATH, 'wb') as outfile:
            pickle.dump(self.raw_data, outfile, pickle.HIGHEST_PROTOCOL)
        print('-> Data() object is saved.\n')
        return

    # LOADS DATA FROM raw_data.pkl INTO self.raw_data
    def load(self):
        with open(RAW_DATAPATH, 'rb') as infile:
            self.raw_data   = pickle.load(infile)
        print('-> Data() object is loaded.\n')
        return
