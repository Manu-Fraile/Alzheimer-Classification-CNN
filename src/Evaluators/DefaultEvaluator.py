import numpy as np


def EvaluateModel(model, x_test, y_test, nclasses=2):
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    print('+++++   BEST MODEL   +++++')
    print('The test loss: ' + str(test_loss) + '\nThe test accuracy: ' + str(test_acc))
    preds = model.predict(x_test)
    label_preds = np.argmax(preds, axis=1)

    # for precision and rest
    one_hot_labels = np.zeros((len(y_test), nclasses))
    for i in range(len(y_test)):
        one_hot_labels[i, label_preds[i]] = 1

    for j in range(nclasses):
        print(f'Eval for class {j}')
        ExtraMetric(one_hot_labels[:, j], y_test[:, j])
        print('')


def ExtraMetric(predicted, true):
    TP = np.abs(np.sum(predicted * true))
    TN = np.abs(np.sum((predicted - 1) * (true - 1)))
    FP = np.abs(np.sum(predicted * (true - 1)))
    FN = np.abs(np.sum((predicted - 1) * true))

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)

    print(f'Precision: {precision}')
    print(f'Recall:    {recall}')
    print(f'f1-score:  {f1}')

    # return precision, recall, f1
