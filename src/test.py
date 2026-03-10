from sklearn.metrics import auc, roc_curve, confusion_matrix, precision_recall_curve, average_precision_score
import numpy as np
import torch
from src.utils import chunk_to_label, gt_complete_to_gt_chunk

def cal_false_alarm(gt, preds, threshold=0.5):
    preds = preds.cpu().detach().numpy()
    gt = gt.cpu().detach().numpy()

    # preds = np.repeat(preds, 16)
    preds[preds < threshold] = 0
    preds[preds >= threshold] = 1
    tn, fp, fn, tp = confusion_matrix(gt, preds, labels=[0, 1]).ravel()

    far = fp / (fp + tn)

    return far

def calculate_average_recall(ground_truth, probs, num_thresholds=11, probabilities=False):
    """
    Calculates the mean Recall across a range of thresholds [0, 1].
    """
    gt = np.array(ground_truth)
    # Convert logits to probabilities between 0 and 1
    if not probabilities:
        probs = 1 / (1 + np.exp(-np.array(probs))) 
    
    thresholds = np.linspace(0, 1, num_thresholds)
    recalls = []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        
        # Recall = TP / (TP + FN)
        tp = np.sum((preds == 1) & (gt == 1))
        actual_positives = np.sum(gt == 1)
        
        recall = tp / actual_positives if actual_positives > 0 else 0.0
        recalls.append(recall)
    
    return np.mean(recalls)


def calculate_average_precision(ground_truth, probs, num_thresholds=11, probabilities=False):
    """
    Calculates the mean Precision across a range of thresholds [0, 1].
    """
    gt = np.array(ground_truth)
    # Convert logits to probabilities between 0 and 1
    if not probabilities:
        probs = 1 / (1 + np.exp(-np.array(probs))) 
    
    thresholds = np.linspace(0, 1, num_thresholds)
    precisions = []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        
        # Precision = TP / (TP + FP)
        tp = np.sum((preds == 1) & (gt == 1))
        predicted_positives = np.sum(preds == 1)
        
        precision = tp / predicted_positives if predicted_positives > 0 else 0.0
        precisions.append(precision)
    
    return np.mean(precisions)


def test_func(dataloader, model, gt, dataset, model_mode):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0).cuda()
        abnormal_preds = torch.zeros(0).cuda()
        abnormal_labels = torch.zeros(0).cuda()
        normal_preds = torch.zeros(0).cuda()
        normal_labels = torch.zeros(0).cuda()
        # Force it to be a long (integer) tensor
        gt_tmp = torch.tensor(gt.copy(), dtype=torch.long).cuda()

        for i, (v_input, label, t_feat) in enumerate(dataloader):
            v_input = v_input.float().cuda(non_blocking=True)
            seq_len = torch.sum(torch.max(torch.abs(v_input), dim=2)[0] > 0, 1)

            logits, _ = model(v_input, seq_len, t_feat)
            logits = torch.mean(logits, 0)
            pred = torch.cat((pred, logits))
            labels = chunk_to_label(gt_tmp[: seq_len[0] * 16], seq_len[0], 16)
            if torch.sum(labels) == 0:
                normal_labels = torch.cat((normal_labels, labels))
                normal_preds = torch.cat((normal_preds, logits))
            else:
                abnormal_labels = torch.cat((abnormal_labels, labels))
                abnormal_preds = torch.cat((abnormal_preds, logits))
            gt_tmp = gt_tmp[seq_len[0] * 16:]

        pred = pred.cpu().detach().numpy().flatten()
        pred = list(pred)
        n_far = cal_false_alarm(normal_labels, normal_preds)
        fpr, tpr, _ = roc_curve(list(gt_complete_to_gt_chunk(gt, 16)), pred)
        roc_auc = auc(fpr, tpr)
        pre, rec, _ = precision_recall_curve(list(gt_complete_to_gt_chunk(gt, 16)), pred)
        pr_auc = calculate_average_precision(list(gt_complete_to_gt_chunk(gt, 16)), pred, probabilities=model_mode==4)
        arec = calculate_average_recall(list(gt_complete_to_gt_chunk(gt, 16)), pred, probabilities=model_mode==4)
        # arec, _ = compute_average_recall(list(gt), np.repeat(pred, 16))

        if dataset == 'ucf-crime':
            return roc_auc, n_far
        elif dataset == 'xd-violence':
            return pr_auc, n_far
        elif dataset == 'shanghaiTech':
            return roc_auc, n_far
        elif dataset == 'ucf2':
            return roc_auc, n_far
        elif dataset == 'bridge':
            return pr_auc, arec
        elif dataset == 'ordering':
            return pr_auc, arec
        elif dataset == 'mutex':
            return pr_auc, arec
        else:
            raise RuntimeError('Invalid dataset.')
