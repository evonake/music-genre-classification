import torch
torch.manual_seed(123)
from torch.autograd import Variable

from config import GENRES, DATAPATH, MODELPATH
from model  import genreNet
from data   import Data
from set    import Set

import matplotlib.pyplot as plt
import torchvision.models as models


def main():
    # DATA -------------------------------------------------------------------------------------- #
    # data = Data(GENRES, DATAPATH)
    # data.make_raw_data()
    # data.save()
    data = Data(GENRES, DATAPATH)
    data.load()
    # ------------------------------------------------------------------------------------------- #
    # SET --------------------------------------------------------------------------------------- #
    # set_ = Set(data)
    # set_.make_dataset()
    # set_.save()
    set_ = Set(data)
    set_.load()

    x_train, y_train    = set_.get_train_set()
    x_valid, y_valid    = set_.get_valid_set()
    x_test,  y_test     = set_.get_test_set()
    # ------------------------------------------------------------------------------------------- #

    TRAIN_SIZE  = len(x_train)
    VALID_SIZE  = len(x_valid)
    TEST_SIZE   = len(x_test)

    net = models.ResNet34()
    net.cuda()

    criterion   = torch.nn.CrossEntropyLoss()
    optimizer   = torch.optim.Adam(net.parameters(), lr=1e-3)

    EPOCH_NUM   = 300
    BATCH_SIZE  = 10

    # LIVE GRAPH SETUP # LIVE GRAPH SETUP # LIVE GRAPH SETUP # LIVE GRAPH SETUP # LIVE GRAPH SETUP -------- #
    fig = plt.figure(figsize=(6.8, 7.6))
    acc_fig  = fig.add_subplot(211)
    loss_fig = fig.add_subplot(212)

    acc_fig.set_ylabel('Accuracy')
    acc_fig.set_xlim(0, EPOCH_NUM)
    acc_fig.set_ylim(0, 1)

    loss_fig.set_xlabel('Epoch')
    loss_fig.set_xlim(0, EPOCH_NUM)
    loss_fig.set_ylabel('Loss')
    loss_fig.set_ylim(0, 3)

    # TRAIN GRAPH COLOR: BLUE --- VALID GRAPH COLOR: RED
    TRAIN_COLOR, VALID_COLOR = '#0000FF', '#FF0000'
    p1, p2 = None, None

    train_acc_hist, train_loss_hist, valid_acc_hist, valid_loss_hist  = [], [], [], []
    # ----------------------------------------------------------------------------------------------------- #

    # MISC METADATA #
    last_train_loss, last_valid_loss, last_train_acc, last_valid_acc = 3, 3, 0, 0
    # ------------- #

    inp_train, out_train    = Variable(torch.from_numpy(x_train)).float().cuda(), Variable(torch.from_numpy(y_train)).long().cuda()
    inp_valid, out_valid    = Variable(torch.from_numpy(x_valid)).float().cuda(), Variable(torch.from_numpy(y_valid)).long().cuda()

    for epoch in range(EPOCH_NUM):
        # ------------------------------------------------------------------------------------------------- #
        # ADAPTIVE LEARNING # ADAPTIVE LEARNING # ADAPTIVE LEARNING # ADAPTIVE LEARNING # ADAPTIVE LEARNING #
        # if epoch == 200:
            # for param_group in optimizer.param_groups:
                # param_group['lr'] = 1e-4
        # ------------------------------------------------------------------------------------------------- #
        # ------------------------------------------------------------------------------------------------- #
        ## TRAIN PHASE # TRAIN PHASE # TRAIN PHASE # TRAIN PHASE # TRAIN PHASE # TRAIN PHASE # TRAIN PHASE ##
        net.train()
        # ------------------------------------------------------------------------------------------------- #
        train_loss = 0
        optimizer.zero_grad()  # <-- OPTIMIZER
        for i in range(0, TRAIN_SIZE, BATCH_SIZE):
            x_train_batch, y_train_batch = inp_train[i:i + BATCH_SIZE], out_train[i:i + BATCH_SIZE]

            pred_train_batch    = net(x_train_batch)
            loss_train_batch    = criterion(pred_train_batch, y_train_batch)
            train_loss          += loss_train_batch.data.cpu().numpy()

            loss_train_batch.backward()
        optimizer.step()  # <-- OPTIMIZER

        epoch_train_loss    = (train_loss * BATCH_SIZE) / TRAIN_SIZE
        train_sum           = 0
        for i in range(0, TRAIN_SIZE, BATCH_SIZE):
            pred_train      = net(inp_train[i:i + BATCH_SIZE])
            indices_train   = pred_train.max(1)[1]
            train_sum       += (indices_train == out_train[i:i + BATCH_SIZE]).sum().data.cpu().numpy()
        train_accuracy  = train_sum / float(TRAIN_SIZE)

        # ------------------------------------------------------------------------------------------------- #
        ## VALIDATION PHASE ## VALIDATION PHASE ## VALIDATION PHASE ## VALIDATION PHASE ## VALIDATION PHASE #
        net.eval()
        # ------------------------------------------------------------------------------------------------- #
        with torch.no_grad():
            valid_loss = 0
            for i in range(0, VALID_SIZE, BATCH_SIZE):
                x_valid_batch, y_valid_batch = inp_valid[i:i + BATCH_SIZE], out_valid[i:i + BATCH_SIZE]

                pred_valid_batch    = net(x_valid_batch)
                loss_valid_batch    = criterion(pred_valid_batch, y_valid_batch)
                valid_loss          += loss_valid_batch.data.cpu().numpy()

            epoch_valid_loss    = (valid_loss * BATCH_SIZE) / VALID_SIZE
            valid_sum           = 0
            for i in range(0, VALID_SIZE, BATCH_SIZE):
                pred_valid      = net(inp_valid[i:i + BATCH_SIZE])
                indices_valid   = pred_valid.max(1)[1]
                valid_sum       += (indices_valid == out_valid[i:i + BATCH_SIZE]).sum().data.cpu().numpy()
            valid_accuracy  = valid_sum / float(VALID_SIZE)

        ## POST-TRAIN/VALID PHASE GRAPH # POST-TRAIN/VALID PHASE GRAPH # POST-TRAIN/VALID PHASE GRAPH ----- #
        if epoch % 5 == 0 or epoch == EPOCH_NUM - 1:
            epoch_hist = [i for i in range(0, epoch + 2, 5)]
            train_acc_hist.append(train_accuracy)
            train_loss_hist.append(epoch_train_loss)
            valid_acc_hist.append(valid_accuracy)
            valid_loss_hist.append(epoch_valid_loss)

            acc_fig.plot(epoch_hist, train_acc_hist,  c=TRAIN_COLOR)
            loss_fig.plot(epoch_hist, train_loss_hist, c=TRAIN_COLOR)
            acc_fig.plot(epoch_hist, valid_acc_hist,  c=VALID_COLOR)
            loss_fig.plot(epoch_hist, valid_loss_hist, c=VALID_COLOR)
            plt.pause(0.1)
        # ------------------------------------------------------------------------------------------------- #
        print((
            f'e: {epoch + 1}\t'
            f't_loss : {"%.4f" % epoch_train_loss}{" ˄" if epoch_train_loss > last_train_loss else "  ˅"}\t'
            f'v_loss : {"%.4f" % epoch_valid_loss}{" ˄" if epoch_valid_loss > last_valid_loss else "  ˅"}\t'
            f't_acc : {"%.2f" % (train_accuracy * 100)}{" ˄" if train_accuracy * 100 > last_train_acc else "  ˅"}  \t'
            f'v_acc : {"%.2f" % (valid_accuracy * 100)}{" ˄" if valid_accuracy * 100 > last_valid_acc else "  ˅"}'))
        last_train_loss, last_valid_loss, last_train_acc, last_valid_acc = epoch_train_loss, epoch_valid_loss, train_accuracy * 100, valid_accuracy * 100
        # ------------------------------------------------------------------------------------------------- #

    plt.show()

    # ------------------------------------------------------------------------------------------------- #
    ## FINAL MESSAGE
    # ------------------------------------------------------------------------------------------------- #
    # print('with dropout(.2), batchnorm, 1e-3 (200) -> 1e-4 (500), batch=16')

    # ------------------------------------------------------------------------------------------------- #
    ## SAVE GENRENET MODEL
    # ------------------------------------------------------------------------------------------------- #
    torch.save(net.state_dict(), MODELPATH)
    print('-> ptorch model is saved.')
    # ------------------------------------------------------------------------------------------------- #

    # ------------------------------------------------------------------------------------------------- #
    ## EVALUATE TEST ACCURACY
    net.eval()
    # ------------------------------------------------------------------------------------------------- #
    inp_test, out_test = Variable(torch.from_numpy(x_test)).float().cuda(), Variable(torch.from_numpy(y_test)).long().cuda()
    test_sum = 0
    for i in range(0, TEST_SIZE, BATCH_SIZE):
        pred_test       = net(inp_test[i:i + BATCH_SIZE])
        indices_test    = pred_test.max(1)[1]
        test_sum        += (indices_test == out_test[i:i + BATCH_SIZE]).sum().data.cpu().numpy()
    test_accuracy   = test_sum / float(TEST_SIZE)
    print('Test acc : %.2f\t' % float(test_accuracy * 100), test_sum, '/', TEST_SIZE)
    # ------------------------------------------------------------------------------------------------- #

    return


if __name__ == '__main__':
    main()
