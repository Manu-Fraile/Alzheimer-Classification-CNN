import numpy as np

import src.DataPlotting.AccuracyPlots as AccuracyPlots
import src.DataPlotting.LossPlots as LossPlots


def SaveResults(history, model, experimentRoute, x_test, y_test):
    AccuracyPlots.PlotAccuracy(history, experimentRoute)
    LossPlots.PlotLoss(history, experimentRoute)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    testf = open(experimentRoute + 'results.txt', 'w')
    testf.write('+++++   BEST MODEL   +++++')
    testf.write('\nThe test loss: ' + str(test_loss) + '\nThe test accuracy: ' + str(test_acc))
    testf.close()

    preds = model.predict(x_test)
    probsf = open(experimentRoute + 'probabilities.txt', 'w')
    probsf.write('The probability labels are:\n')
    probsf.write(str(preds))
    probsf.close()

    label_preds = np.argmax(preds, axis=1)
    predsf = open(experimentRoute + 'predictions.txt', 'w')
    predsf.write('The predicted labels are:\n')
    predsf.write(str(label_preds))
    predsf.close()
