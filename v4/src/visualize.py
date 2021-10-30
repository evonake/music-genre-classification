from config import GENRES, DATAPATH, MODELPATH
from model  import genreNet
from data   import Data
from set    import Set

import random
import librosa.display
import matplotlib.pyplot as plt


def main():
    data = Data(GENRES, DATAPATH)
    data.load()

    set_ = Set(data)
    set_.load()

    x_set, y_set = set_.get_train_set()

    fig = plt.figure(figsize=(16, 8))

    c = random.randint(0, 6000)
    for i in range(10):
        subplt = fig.add_subplot(2, 5, i + 1)

        while y_set[c] != i:
            c += 1
        entry, y_entry = x_set[c].squeeze(), GENRES[y_set[c]]
        img = librosa.display.specshow(entry, fmax=8000)

        subplt.set_xlabel(y_entry)

    plt.show()


if __name__ == '__main__':
    main()
