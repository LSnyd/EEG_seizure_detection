import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score



#Give out AUC and print ROC curve

def auc(true,pred):

    print("AUC score", roc_auc_score(true, pred))



def rocplot(true,pred):
    fpr, tpr, _ = roc_curve(true, pred)

    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.margins(0.5)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.plot(fpr, tpr, 'b')
    plt.show()
