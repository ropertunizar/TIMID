
import time
from src.utils import fixed_smooth, slide_smooth
from src.test import *
from src.visualizer import visualize

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


def calculate_f1(ground_truth, probs, num_thresholds=11, probabilities=False):
    """
    Calculates the mean F1 score across a range of thresholds [0, 1].
    """
    gt = np.array(ground_truth)
    # Convert logits to probabilities between 0 and 1
    if not probabilities:
        probs = 1 / (1 + np.exp(-np.array(probs))) 
    
    thresholds = np.linspace(0, 1, num_thresholds)
    f1_scores = []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        
        # F1 = 2 * (Precision * Recall) / (Precision + Recall)
        tp = np.sum((preds == 1) & (gt == 1))
        predicted_positives = np.sum(preds == 1)
        actual_positives = np.sum(gt == 1)

        precision = tp / predicted_positives if predicted_positives > 0 else 0.0
        recall = tp / actual_positives if actual_positives > 0 else 0.0

        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
        print(f"Threshold: {t:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        f1_scores.append(f1)

    return np.mean(f1_scores)


def infer_func(model, dataloader, gt, logger, cfg, model_mode):
    st = time.time()
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0).cuda()
        normal_preds = torch.zeros(0).cuda()
        normal_labels = torch.zeros(0).cuda()
        normal_labels_chunk = torch.zeros(0).cuda()
        gt_tmp = torch.tensor(gt.copy(), dtype=torch.long).cuda()

        for i, (v_input, name, t_feat) in enumerate(dataloader):
            # print(f'Processing video {name}')
            v_input = v_input.float().cuda(non_blocking=True)
            seq_len = torch.sum(torch.max(torch.abs(v_input), dim=2)[0] > 0, 1)
            logits, _ = model(v_input, seq_len, t_feat)
            logits = torch.mean(logits, 0)
            logits = logits.squeeze(dim=-1)

            seq = len(logits)
            if cfg.smooth == 'fixed':
                logits = fixed_smooth(logits, cfg.kappa)
            elif cfg.smooth == 'slide':
                logits = slide_smooth(logits, cfg.kappa)
            else:
                pass
            logits = logits[:seq]

            pred = torch.cat((pred, logits))
            labels = gt_tmp[: seq_len[0]*16]
            if torch.sum(labels) == 0:
                normal_labels = torch.cat((normal_labels, labels))
                normal_preds = torch.cat((normal_preds, logits))
                normal_labels_chunk = torch.cat((normal_labels_chunk, chunk_to_label(labels, seq_len[0], 16)))
            gt_tmp = gt_tmp[seq_len[0]*16:]

            if cfg.plot:
                # print(name[0])
                visualize(name[0], np.repeat(logits.cpu().detach().numpy(), 16), labels)

            # probs = 1 / (1 + np.exp(-logits.cpu().detach().numpy()))
            # save probabilities 
            # np.save(f'./probs/pel4vad/{name[0]}.npy', logits.cpu().detach().numpy())


        pred = list(pred.cpu().detach().numpy())
        far = cal_false_alarm(normal_labels_chunk, normal_preds)
        fpr, tpr, _ = roc_curve(list(gt_complete_to_gt_chunk(gt, 16)), pred)
        roc_auc = auc(fpr, tpr)
        # pre, rec, _ = precision_recall_curve(list(gt_complete_to_gt_chunk(gt, 16)), pred)
        pr_auc = calculate_average_precision(list(gt_complete_to_gt_chunk(gt, 16)), pred, model_mode==4)
        arec = calculate_average_recall(list(gt_complete_to_gt_chunk(gt, 16)), pred, model_mode==4)

        # add F1
        f1 = calculate_f1(list(gt_complete_to_gt_chunk(gt, 16)), pred, model_mode==4)

    time_elapsed = time.time() - st
    logger.info('offline AUC:{:.4f} AP:{:.4f} FAR:{:.4f} AR:{:.4f} F1:{:.4f} | Complete in {:.0f}m {:.0f}s\n'.format(
        roc_auc, pr_auc, far, arec, np.mean(f1), time_elapsed // 60, time_elapsed % 60))