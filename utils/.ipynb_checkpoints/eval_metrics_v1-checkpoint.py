from torchmetrics import IoU
from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_score
import numpy as np

# import train

# Utils
LABELS=['No Buildings','Buildings']

def metrics(predictions, gts, label_values=LABELS):
    cm = confusion_matrix(gts,predictions,labels=range(len(label_values)))
    print("Confusion matrix :")
    print(cm)
    print("------------------------------------")
    
    # Compute global accuracy
    total = sum(sum(cm))
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    print("{} pixels processed".format(total))
    print("Total accuracy : {}%".format(accuracy))
    
    print("------------------------------------")
    
    # Compute F1 score
    F1Score = np.zeros(len(label_values))
    for i in range(len(label_values)):
        try:
            F1Score[i] = 2. * cm[i,i] / (np.sum(cm[i,:]) + np.sum(cm[:,i]))
        except:
            # Ignore exception if there is no element in class i for test set
            pass
    print("F1Score :")
    for l_id, score in enumerate(F1Score):
        print("{}: {}".format(label_values[l_id], score))

    print("----------------------------")
        
    # Compute kappa coefficient
    total = np.sum(cm)
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total*total)
    kappa = (pa - pe) / (1 - pe);
    print("Kappa: " + str(kappa))
    
    print("----------------------------")
    
    #compute jaccard distance/ for similarity it is 1-J
    """intersection is equivalent to True Positive count
    union is the mutually inclusive area of all labels & predictions"""
    smooth = 1e-7
    intersection = (predictions * gts).sum()
    total = (predictions + gts).sum()
    union = total - intersection
    IoU = (intersection + smooth) / (union + smooth)
    print(f"Manually computed Jaccard Index:{1-IoU}")
    
    #nxt method
    j_index = jaccard_score(y_true=gts,y_pred=predictions, average='micro')
    print(f"Jaccard Index is:{j_index}")

    return accuracy,kappa,j_index

if __name__ == "__main__":
    metrics()