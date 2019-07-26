from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def draw_ROC(pred, label, show=False, save_path='roc.jpg'):
    fpr, tpr, _ = roc_curve(label, pred)
    roc_auc = auc(fpr, tpr)

    plt.title('ROC Validation')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    if show:
        plt.show()
    plt.savefig(save_path)
