from sklearn.metrics import accuracy_score, confusion_matrix

def calculate_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)
