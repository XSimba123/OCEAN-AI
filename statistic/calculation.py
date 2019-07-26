from sklearn import metrics


def false_alarm_rate(pred, label, show=False):
    thred = 0.0
    threds = []
    fprs = []
    while thred < 1:
        thred += 0.01
        bool_pred = [p > thred for p in pred]
        label_idx = 0
        tp, tn, fp, fn = 0, 0, 0, 0
        for p in bool_pred:
            c = p
            if c == 1 and label[label_idx] == 1:
                tp += 1.0
            elif c == 1 and label[label_idx] == 0:
                fp += 1.0
            elif c == 0 and label[label_idx] == 0:
                tn += 1.0
            elif c == 0 and label[label_idx] == 1:
                fn += 1.0
            label_idx += 1
        fprs.append(fp / (fp + tn))
        threds.append(thred)
    fpr_thred = [0.1, 0.2, 0.3, 0.4, 0.5]
    min_threds = []
    _ft = 0
    for ft in fpr_thred:
        min_fpr, min_thred = 0, 1
        for fpr, t in zip(fprs, threds):
            if _ft < fpr < ft and fpr > min_fpr:
                min_fpr = fpr
                min_thred = t
        _ft = ft
        if show:
            print('False alarm rate under %f, best fpr: %f, best thred: %f' % (ft, min_fpr, min_thred))
        min_threds.append(min_thred)
    return min_threds


def confusion_matrix(pred, labels, show_mat=False):
    tp, tn, fp, fn = 0, 0, 0, 0
    res = []
    for p, label in zip(pred, labels):
        if label == p['classes'] == 0:
            tn += 1
        elif label == p['classes'] == 1:
            tp += 1
        elif label == 0 and p['classes'] == 1:
            fp += 1
        elif label == 1 and p['classes'] == 0:
            fn += 1
        if show_mat:
            pass
            # print(p)
        res.append([p['classes'], label, p['probabilities'][0], p['probabilities'][1]])
    sens = float(tp) / (tp + fn)
    spec = float(tn) / (fp + tn)
    acc = float(tp + tn) / (tp + fp + tn + fn)
    if show_mat:
        print('TP: %i, TN: %i, FP: %i, FN: %i' % (tp, tn, fp, fn))
        print('Sensitvity: %f, Specifity: %f, Accuracy: %f' % (sens, spec, acc))
    return sens, spec, acc


def AUC(pred, labels, show=False):
    prob = [p['probabilities'][1] for p in pred]

    fpr, tpr, thresh = metrics.roc_curve(labels, prob)
    auc = metrics.auc(fpr, tpr)
    if show:
        print("AUC is %s" % auc)
    return auc
