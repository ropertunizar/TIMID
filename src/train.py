import torch
from src.loss import *
from src.utils import *
from infonce import SupervisedInfoNCE
from pytorch_metric_learning import losses


def train_func(dataloader, model, optimizer, criterion, criterion2, model_mode, lamda=0):
    t_loss = []
    s_loss = []

    loss_func = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)

    with torch.set_grad_enabled(True):
        model.train()
        for i, (v_input, t_input, label, multi_label) in enumerate(dataloader):
            seq_len = torch.sum(torch.max(torch.abs(v_input), dim=2)[0] > 0, 1)
            v_input = v_input[:, :torch.max(seq_len), :]
            v_input = v_input.float().cuda(non_blocking=True)
            t_input = t_input.float().cuda(non_blocking=True)
            label = label.float().cuda(non_blocking=True)
            multi_label = multi_label.cuda(non_blocking=True)

            logits, v_feat = model(v_input, seq_len, t_input)

            # Prompt-Enhanced Learning
            if model_mode == 4:
                logit_scale = model.logit_scale.exp()
                video_feat, token_feat, video_labels = get_cas(v_feat, t_input, logits, multi_label)
                v2t_logits, v2v_logits = create_logits(video_feat, token_feat, logit_scale)
                ground_truth = torch.tensor(gen_label(video_labels), dtype=v_feat.dtype).cuda()
                loss2 = KLV_loss(v2t_logits, ground_truth, criterion2)

            # change 0's to -1
            contrastive_labels = torch.where(multi_label == 0, torch.tensor(-1.0).cuda(), multi_label)

            loss1 = CLAS2(logits, label, seq_len, criterion) if model_mode != 4 else CLAS2_p4v(logits, label, seq_len, criterion)
            if model_mode != 4:
                pooled_feats = []
                
                for b in range(v_feat.shape[0]):
       
                    valid_frames = v_feat[b, :seq_len[b], :] 
                    
                    pooled_video = torch.mean(valid_frames, dim=0) 
                    pooled_feats.append(pooled_video)
                    
                v_feat_pooled = torch.stack(pooled_feats)

          
                loss2 = loss_func(v_feat_pooled, contrastive_labels)
          
            loss = loss1 + lamda * loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t_loss.append(loss1)
            s_loss.append(loss2)

    return sum(t_loss) / len(t_loss), sum(s_loss) / len(s_loss)
