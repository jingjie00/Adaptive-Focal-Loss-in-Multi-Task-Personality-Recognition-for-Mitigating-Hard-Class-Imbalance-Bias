def e_confusion_matrix(preds,labels):
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for pred, label in zip(preds, labels):
        if pred == 1 and label == 1:
            true_positive += 1
        elif pred == 1 and label == 0:
            false_positive += 1
        elif pred == 0 and label == 0:
            true_negative += 1
        elif pred == 0 and label == 1:
            false_negative += 1

    return true_positive, false_positive, true_negative, false_negative

def e_accuracy(confusion_matrix):
    true_positive, false_positive, true_negative, false_negative = confusion_matrix

    negative = true_negative + false_positive
    positive = true_positive + false_negative

    if negative == 0 or positive == 0:
        return 0,0
        
    sensitivity = true_positive / positive
    specificity = true_negative / negative

    regular_accuracy = (true_negative + true_positive)/(true_positive + false_negative + true_negative + false_positive)
    balanced_accuracy = (sensitivity + specificity) / 2

    # assert positive == 0 or negative == 0, "There is no positive or negative in the dataset"
    
    return regular_accuracy, balanced_accuracy



def evaluation_result(preds,labels):
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for pred, label in zip(preds, labels):
        if pred == 1 and label == 1:
            true_positive += 1
        elif pred == 1 and label == 0:
            false_positive += 1
        elif pred == 0 and label == 0:
            true_negative += 1
        elif pred == 0 and label == 1:
            false_negative += 1

    negative = true_negative + false_positive
    positive = true_positive + false_negative

        
    sensitivity = true_positive / positive
    specificity = true_negative / negative

    regular_accuracy = (true_negative + true_positive)/(true_positive + false_negative + true_negative + false_positive)
    balanced_accuracy = (sensitivity + specificity) / 2
    cm =[[true_positive,true_negative],[false_positive, false_negative]]
    
    #print("RA:",regular_accuracy)
    #print("BA:",balanced_accuracy)
    #print("Confussion matrix: TP,FP,TN,FN", cm)
    assert positive == 0 or negative == 0, "There is no positive or negative in the dataset"
    assert len(preds) == len(labels), "The length of the prediction and the labels are not the same"
    
    return regular_accuracy, balanced_accuracy, cm

def test():
    print("success")