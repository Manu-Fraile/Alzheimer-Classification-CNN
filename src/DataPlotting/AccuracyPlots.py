import matplotlib.pyplot as plt
import src.Router.RouteChecks as RouteChecks


def PlotAccuracy(history, experimentRoute):
    plt.figure(1)
    plt.grid(True)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(bottom=0)
    plt.legend(loc='lower right')

    RouteChecks.CheckRoute(experimentRoute + '/plots/')
    plt.savefig(experimentRoute + 'plots/accuracy.png')
