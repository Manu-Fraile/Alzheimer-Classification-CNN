import matplotlib.pyplot as plt
import src.Router.RouteChecks as RouteChecks


def PlotLoss(history, experimentRoute):
    plt.figure(2)
    plt.grid(True)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(bottom=0)
    plt.legend(loc='lower right')

    RouteChecks.CheckRoute(experimentRoute + '/plots')
    plt.savefig(experimentRoute + '/plots/loss.png')
