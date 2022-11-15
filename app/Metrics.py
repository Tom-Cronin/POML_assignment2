def accuracy(predictions, labels):
    right = 0
    wrong = 0
    for index in range(len(predictions)):
        if predictions[index] == labels[index]:
            right +=1
        else:
            wrong +=1
    return right/(wrong+right)

def confusion_matrix(preds, labels):
    confusion_matrix = [[0, 0], [0, 0]]
    for index in range(len(preds)):
        if preds[index] == labels[index]:
            if preds[index] == 1:  # True positive
                confusion_matrix[0][0] += 1
            else:
                confusion_matrix[1][1] += 1  # True negative
        else:
            if preds[index] == 1:
                confusion_matrix[0][1] += 1  # False Positive
            else:
                confusion_matrix[1][0] += 1  # False Negative
    return confusion_matrix

def precision(preds, labels):
    cf = confusion_matrix(preds,labels)
    TP = cf[0][0]   # True Positive
    FP = cf[0][1]   # True Negative
    return TP / (TP + FP)

def recall(preds, labels):
    cf = confusion_matrix(preds,labels)
    TP = cf[0][0]   # True Positive
    FN = cf[1][0]   # False Negative
    return TP / (TP + FN)

def f1_score(preds, labels):
    prec = precision(preds, labels)
    rec = recall(preds, labels)
    f1 = 2 * (prec * rec)/(prec + rec)
    return f1