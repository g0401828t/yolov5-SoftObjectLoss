# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""                
import pdb

import torch
import torch.nn as nn

import numpy as np

from utils.metrics import bbox_iou
from utils.torch_utils import is_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, loss_type, n, gamma=1, sigma=1, autobalance=False):
        self.sort_obj_iou = False
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        self.NegBCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0], device=device))
        self.sigmoid = nn.Sigmoid()

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

        self.loss_type = loss_type
        self.n = n
        self.gamma = gamma
        self.sigma = sigma

    def __call__(self, p, targets, epoch):  # predictions, targets, model
        # ==== shapes === #
        # p : list of len 3 scales
        # p[0].shape =  [N, 3, 80, 80, 85]
        # p[1].shape =  [N, 3, 40, 40, 85]
        # p[2].shape =  [N, 3, 20, 20, 85]   
        # 85 => 1 (objectness) + 4 (x, y, w, h) + 80 (classes)
        # 
        # indices, tcls, tbox, anchors : list of len 3 scales
        # indices[0] : tuple of 4 (image_idx, anchor_idx, grid_y, grid_x) each shape of [num_targets]
        # tcls[0].shape = [num_targets]
        # tbox[0].shape = [num_targets, 4]
        # anchors[0].shape = [num_targets, 2]
        # =============== #
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        """     custom building targets     """
        # tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets
        # tcls, tbox, indices, anchors, soft_indices = self.build_targets_custom(p, targets)  # targets
        tcls, tbox, indices, anchors, soft_indices = self.build_targets_custom1(p, targets, epoch)  # targets
        """================================="""

        # pure_ious = torch.zeros(1, device=device)

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions (Scale)
            # pdb.set_trace()
            # print("===============>", i)
            # print(soft_indices[i].shape)
            b, a, gj, gi = indices[i]   # image, anchor, gridy, gridx
            """     soft labels     """
            b_soft, a_soft, gj_soft, gi_soft, score = soft_indices[i]   # image, anchor, gridy, gridx
            # pdb.set_trace()
            # tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj [N, 3, S, S]
            # tobj = torch.zeros_like(pi[..., 0], device=device, dtype=torch.half)  # target obj [N, 3, S, S]
            tobj = torch.zeros_like(pi[..., 0], device=device, dtype=torch.double)  # target obj [N, 3, S, S]

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # ps.shape = [num_targets, 85] / prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)   ### <====== bbox regression loss
                lbox += (1.0 - iou).mean()  # iou loss
                ## custom
                # iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, customIoU=True)  # iou(prediction, target)   ### <======= custom bbox regression loss
                # lbox += iou.mean()  # iou loss

                # pure_iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False)  
                # pure_ious += pure_iou.mean()

                # # Objectness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                # if self.sort_obj_iou:
                #     sort_id = torch.argsort(score_iou)
                #     b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                # tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio [N, 3, S, S] /// (0) is no obj, (iou) is obj
                """     Objectness      """
                tobj[b_soft, a_soft, gj_soft, gi_soft] = score
                tobj[b, a, gj, gi] = 1              # overlap the centerpoint conf score to 1
                # tobj[b_soft, a_soft, gj_soft, gi_soft] = score * score_iou                                                                   # => 12월 23일 score [17541] score_iou [652]로 사이즈 달라서 안됨
                # tobj[b, a, gj, gi] = 1 * score_iou             # overlap the centerpoint conf score to 1


                pos_mask = tobj == 1
                neg_mask = tobj == 0
                # soft_mask = (tobj > 0) * (tobj < 1)
                hard_mask = pos_mask + neg_mask
                soft_mask = hard_mask.logical_not()
                """====================="""


                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            """     Soft Object Loss        """
            # obji = self.BCEobj(pi[..., 4], tobj)
            hard_obji = self.BCEobj(pi[..., 4][hard_mask], tobj[hard_mask])

            # Soft_Object_Loss_1
            if self.loss_type == "loss1":
                soft_obji = self.soft_obj_loss1(pi[..., 4][soft_mask], tobj[soft_mask])
            if self.loss_type == "loss11":
                soft_obji = self.soft_obj_loss11(pi[..., 4][soft_mask], tobj[soft_mask], self.gamma)
            if self.loss_type == "loss2":
                # soft_obji = self.soft_obj_loss2(pi[..., 4][soft_mask], tobj[soft_mask]) - self.soft_obj_loss2(tobj[soft_mask], tobj[soft_mask])       # both pos, neg => close to score
                # soft_obji = self.BCEobj(pi[..., 4][soft_mask], tobj[soft_mask]) - self.BCEobj(tobj[soft_mask], tobj[soft_mask])                       # both pos, neg => close to score
                soft_obji = self.soft_obj_loss22(pi[..., 4][soft_mask], tobj[soft_mask])

            # # Soft_Object_Loss_1
            # soft_obji = self.NegBCEobj(pi[..., 4][soft_mask], tobj[soft_mask])
            # # Soft_Object_Loss_2
            # soft_obji = self.soft_obj_loss2(pi[..., 4][soft_mask], tobj[soft_mask]) - self.soft_obj_loss2(tobj[soft_mask], tobj[soft_mask])       # both pos, neg => close to score
            # soft_obji = self.BCEobj(pi[..., 4][soft_mask], tobj[soft_mask]) - self.BCEobj(tobj[soft_mask], tobj[soft_mask])                       # both pos, neg => close to score
            # soft_obji = self.soft_obj_loss22(pi[..., 4][soft_mask], tobj[soft_mask])
            # # Soft_Object Loss 3
            # soft_obji = self.soft_obj_loss3(pi[..., 4][soft_mask], tobj[soft_mask])

            # soft_obji = soft_obji.mean()
            obji = hard_obji + soft_obji
            # obji *= torch.tensor(0.5)
            """============================="""
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        # lobj *= self.hyp['obj'] * (1/6)
        # lobj *= self.hyp['obj'] * (1/3)
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        # pure_ious *= self.hyp['box']
        bs = tobj.shape[0]  # batch size

        # return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls, pure_ious)).detach()
        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()
    
    def soft_obj_loss1(self, p, t):
        p = self.sigmoid(p)
        neg_bce = -(1-t) * torch.log(1-p + 1e-6)
        return neg_bce.mean()

    def soft_obj_loss11(self, p, t, gamma):
        p = self.sigmoid(p)
        neg_bce = -((1-t) ** gamma) * torch.log(1-p + 1e-6)
        return neg_bce.mean()
        
    # def soft_obj_loss2(self, p, t):
    #     p = self.sigmoid(p)
    #     pos_bce = -t * torch.log(p + 1e-6)
    #     neg_bce = -(1-t) * torch.log(1-p + 1e-6)
        
    #     return (pos_bce + neg_bce).mean()

    def soft_obj_loss22(self, p, t):
        # BCE(s, p)
        p = self.sigmoid(p)
        pos_bce = -t * torch.log(p + 1e-6)
        neg_bce = -(1-t) * torch.log(1-p + 1e-6)
        bce = pos_bce + neg_bce

        # BCE(s, s) => offset
        pos_offset = -t * torch.log(t + 1e-6)
        neg_offset = -(1-t) * torch.log(1-t + 1e-6)
        offset = pos_offset + neg_offset
        return (bce - offset).mean()

    def soft_obj_loss3(self, p, t):
        # BCE(s, p)
        p = self.sigmoid(p)
        pos_bce = -t * torch.log(p + 1e-6)
        neg_bce = -(1-t) * torch.log(1-p + 1e-6)
        bce = pos_bce + neg_bce

        # BCE(s, s) => offset
        pos_offset = -t * torch.log(t + 1e-6)
        neg_offset = -(1-t) * torch.log(1-t + 1e-6)
        offset = pos_offset + neg_offset

        loss = (1-t) * (bce - offset)
        return loss.mean()


    def build_targets(self, p, targets):
        # ====== What's Happening ======= #
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        # targets.shape : (N, 6) 6 -> (image, class, x, y, w, h)
        # na : number of anchors (9)
        # nt : number of targets (not Batch size maybe bbox number)
        # nl : number of detection layers (number of Scale : 3)
        # ai : anchor indices
        # 
        # =======  variables    =============================      Shapes    ======================= #
        # t = targets.repeat(na, 1, 1)                          [na, nt, 6]  6:(image, class, x, y, w, h)
        # a = ai[:,:,None]                                      [na, nt, 1]  1:(anchor_index)
        # targets = torch.cat((t, a), 2).shape                  [na, nt, 7]  7:(image, class, x, y, w, h, anchor_index)
        #
        # gain[2:6] = torch.tensor(p[0].shape)[[3,2,3,2]]       [4]         4:(x, y, x, y) => scale!! x, y = 80 for p[0] /  x, y = 40 for p[1] / x, y = 20 for p[2]
        # ========================== #

        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices


        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):            # for loop for num of scale
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # gain = scale: [1, 1, 1, 1, 1, 1, 1] -> [1, 1, S, S, S, S, 1]

            # Match targets to anchors
            t = targets * gain            # [3, 417, 7] : anchors, targets, (image, class, x, y, w, h, anchor_idx)      # x, y, w, h를 scale 단위로 변환 targets: 0~1  --->  t: 0~scale 단위 
            if nt:
                # Matches / Matchin bbox with anchors
                # 해당 anchor랑 얼마나 비슷한지 게산 후 self.hyp['anchor_t'] 보다 작은 것들만 고른다.
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio [3, 417, 2]
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare [3, 417]
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # 해당 anchor와 어느정도 비슷한 predictions 만 남는다. shape : [417, 7] 7 -> image, class, x, y, w, h, anchor_idx

                # Offsets / Neighboring cell assignment
                gxy = t[:, 2:4]                                                 # grid xy shape: [417, 2]
                gxi = gain[[2, 3]] - gxy                                        # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T                            # j, k shape: [417], [417]
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))               # j는 mask. [기존 cell, lebft, upper, right, lower] 0 or 1 shape:[5, 417]
                t = t.repeat((5, 1, 1))[j]                                      # [417, 7] ->  [5, 417, 7] -> masking(5개 중 최대 3개 선택) -> [1251, 7] 보통 각 cell 당 양옆위아래로 총 3개씩 배정
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]       # [1, 417, 2] + [5, 1, 2] -> [5, 417, 2] -> masking -> [1251, 7]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T                # image, class
            gxy = t[:, 2:4]                         # grid xy
            gwh = t[:, 4:6]                         # grid wh
            gij = (gxy - offsets).long()            # grid cell index
            gi, gj = gij.T                          # grid x,y indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))    # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))                                     # box 해당 cell 에서 gt_point가 얼마나 떨어져있는지
            anch.append(anchors[a])                                                         # anchors
            tcls.append(c)                                                                  # class

        return tcls, tbox, indices, anch


    def build_targets_custom(self, p, targets):
        # ====== What's Happening ======= #
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        # targets.shape : (N, 6) 6 -> (image, class, x, y, w, h)
        # na : number of anchors (9)
        # nt : number of targets (not Batch size maybe bbox number)
        # nl : number of detection layers (number of Scale : 3)
        # ai : anchor indices
        # 
        # =======  variables    =============================      Shapes    ======================= #
        # t = targets.repeat(na, 1, 1)                          [na, nt, 6]  6:(image, class, x, y, w, h)
        # a = ai[:,:,None]                                      [na, nt, 1]  1:(anchor_index)
        # targets = torch.cat((t, a), 2).shape                  [na, nt, 7]  7:(image, class, x, y, w, h, anchor_index)
        #
        # gain[2:6] = torch.tensor(p[0].shape)[[3,2,3,2]]       [4]         4:(x, y, x, y) => scale!! x, y = 80 for p[0] /  x, y = 40 for p[1] / x, y = 20 for p[2]
        # ========================== #

        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        """     Initialize Soft_indices     """
        soft_indices = []
        scale = [80, 40, 20]
        """================================="""
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices


        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):            # for loop for num of scale
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # gain = scale: [1, 1, 1, 1, 1, 1, 1] -> [1, 1, S, S, S, S, 1]

            # Match targets to anchors
            t = targets * gain            # [3, 417, 7] : anchors, targets, (image, class, x, y, w, h, anchor_idx)      # x, y, w, h를 scale 단위로 변환 targets: 0~1  --->  t: 0~scale 단위 
            if nt:
                # Matches / Matchin bbox with anchors
                # 해당 anchor랑 얼마나 비슷한지 게산 후 self.hyp['anchor_t'] 보다 작은 것들만 고른다.
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio [3, 417, 2]
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare [3, 417]
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # 해당 anchor와 어느정도 비슷한 predictions 만 남는다. shape : [417, 7] 7 -> image, class, x, y, w, h, anchor_idx

                # # Offsets / Neighboring cell assignment
                # gxy = t[:, 2:4]                                                 # grid xy shape: [417, 2]
                # gxi = gain[[2, 3]] - gxy                                        # inverse
                # j, k = ((gxy % 1 < g) & (gxy > 1)).T                            # j, k shape: [417], [417]
                # l, m = ((gxi % 1 < g) & (gxi > 1)).T
                # j = torch.stack((torch.ones_like(j), j, k, l, m))               # j는 mask. [기존 cell, lebft, upper, right, lower] 0 or 1 shape:[5, 417]
                # t = t.repeat((5, 1, 1))[j]                                      # [417, 7] ->  [5, 417, 7] -> masking(5개 중 최대 3개 선택) -> [1251, 7] 보통 각 cell 당 양옆위아래로 총 3개씩 배정
                # offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]       # [1, 417, 2] + [5, 1, 2] -> [5, 417, 2] -> masking -> [1251, 7]
            else:
                t = targets[0]
                # offsets = 0

            """     Shapes of Tensors           """
            """ t.shape = [num_targets, 7]      """
            """ gxy.shape = [num_targets, 2]    """
            """================================="""


            # Define
            b, c = t[:, :2].long().T                # image, class
            gxy = t[:, 2:4]                         # grid xy
            gwh = t[:, 4:6]                         # grid wh
            # gij = (gxy - offsets).long()            # grid cell index
            gij = (gxy).long()            # grid cell index
            gi, gj = gij.T                          # grid x,y indices
            a = t[:, 6].long()  # anchor indices
            
            """     Checking cells inside the Bounding Boxes        """
            S = scale[i]
            a_soft = torch.tensor([]).to(targets.device)
            b_soft = torch.tensor([]).to(targets.device)
            gj_soft = torch.tensor([]).to(targets.device)
            gi_soft = torch.tensor([]).to(targets.device)
            score = torch.tensor([]).to(targets.device)
            for target_idx, xy in enumerate(gxy):
                x, y = xy / S
                x, y = x.item(), y.item()
                w, h = gwh[target_idx] / S
                width, height = w.item(), h.item()

                x1, y1 = x - width/2, y - height/2
                x2, y2 = x + width/2, y + height/2
                first_cell_x, first_cell_y = int(x1 * S), int(y1 * S)
                last_cell_x, last_cell_y = int(x2 * S), int(y2 * S)
                if first_cell_x == S:
                    first_cell_x -= 1
                if first_cell_y == S:
                    first_cell_y -= 1
                if last_cell_x == S:
                    last_cell_x -= 1
                if last_cell_y == S:
                    last_cell_y -= 1

                c2 = np.sqrt(width ** 2 + height ** 2)          # bbox의 대각선 길이 => normalize 하려고
                number_of_cells = (last_cell_x + 1 - first_cell_x) * (last_cell_y + 1 - first_cell_y) - 1  # center는 할당 안해준다.
                a_soft = torch.cat((a_soft, a[i].repeat(number_of_cells)), 0)
                b_soft = torch.cat((b_soft, b[i].repeat(number_of_cells)), 0)

                gj_soft_idx = torch.zeros_like(a[i].repeat(number_of_cells)).to(targets.device)
                gi_soft_idx = torch.zeros_like(a[i].repeat(number_of_cells)).to(targets.device)
                score_idx = torch.zeros_like(a[i].repeat(number_of_cells)).to(targets.device)
                idx = 0
                # print("현재 center cell", gi[target_idx], gj[target_idx], "cell 갯수", number_of_cells, "cell idx", first_cell_x, first_cell_y, last_cell_x, last_cell_y)
                for y_cell_idx in range(first_cell_y, last_cell_y+1):
                    for x_cell_idx in range(first_cell_x, last_cell_x+1):
                        # print(x_cell_idx, y_cell_idx)
                        if x_cell_idx != gi[target_idx].clamp_(0, S - 1) or y_cell_idx != gj[target_idx].clamp_(0, S - 1):                           # center가 아닐때만
                            # cell_cx, cell_cy = (x_cell_idx / S) + (1/(2*S)), (y_cell_idx / S) + (1/(2*S))             # 해당 cell의 중간값
                            # distance = torch.sqrt(((x - cell_cx) ** 2) + ((y - cell_cy) ** 2))                    # 해당 cell의 중간값과 center point와의 거리 계산
                            # normalized_distance = distance / c2                                                   # normalized by bbox의 대각선 길이
                            # score = (normalized_distance - 1) ** 30 # 400                                         # 거리에 따른 score 값 부여
                            gi_soft_idx[idx] = x_cell_idx
                            gj_soft_idx[idx] = y_cell_idx
                            # score_idx[idx] = score
                            # print(gi_soft_idx, gj_soft_idx)
                            # pdb.set_trace()
                            idx += 1
                
                cell_cx, cell_cy = (gi_soft_idx / S) + (1/(2*S)), (gj_soft_idx / S) + (1/(2*S))         # 해당 cell의 중간값
                distance = torch.sqrt(((x - cell_cx) ** 2) + ((y - cell_cy) ** 2))                      # 해당 cell의 중간값과 center point와의 거리 계산
                normalized_distance = distance / c2                                                     # normalized by bbox의 대각선 길이
                # score_idx = (normalized_distance - 1) ** 30 # 400                                       # 거리에 따른 score 값 부여
                score_idx = (normalized_distance - 1) ** self.hyp["n"] # 400                          # 거리에 따른 score 값 부여

                # print(number_of_cells)
                # print(first_cell_x, first_cell_y, last_cell_x, last_cell_y)
                # print(distance, c2)
                # print(score_idx)
                # pdb.set_trace()


                gj_soft = torch.cat((gj_soft, gj_soft_idx), 0)
                gi_soft = torch.cat((gi_soft, gi_soft_idx), 0)
                score = torch.cat((score, score_idx), 0)
            """======================================================="""



            # Append
            # a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))    # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))                                     # box 해당 cell 에서 gt_point가 얼마나 떨어져있는지
            anch.append(anchors[a])                                                         # anchors
            tcls.append(c)                                                                  # class

            """     append soft mask and score      """
            a_soft = a_soft.long()
            b_soft = b_soft.long()
            gj_soft = gj_soft.long()
            gi_soft = gi_soft.long()
            # score = score.type(torch.float16)
            score = score.type(torch.double)
            soft_indices.append((b_soft, a_soft, gj_soft.clamp_(0, gain[3] - 1), gi_soft.clamp_(0, gain[2] - 1), score))
            """====================================="""

        return tcls, tbox, indices, anch, soft_indices


    def build_targets_custom1(self, p, targets, epoch):
        # ====== What's Happening ======= #
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        # targets.shape : (N, 6) 6 -> (image, class, x, y, w, h)
        # na : number of anchors (9)
        # nt : number of targets (not Batch size maybe bbox number)
        # nl : number of detection layers (number of Scale : 3)
        # ai : anchor indices
        # 
        # =======  variables    =============================      Shapes    ======================= #
        # t = targets.repeat(na, 1, 1)                          [na, nt, 6]  6:(image, class, x, y, w, h)
        # a = ai[:,:,None]                                      [na, nt, 1]  1:(anchor_index)
        # targets = torch.cat((t, a), 2).shape                  [na, nt, 7]  7:(image, class, x, y, w, h, anchor_index)
        #
        # gain[2:6] = torch.tensor(p[0].shape)[[3,2,3,2]]       [4]         4:(x, y, x, y) => scale!! x, y = 80 for p[0] /  x, y = 40 for p[1] / x, y = 20 for p[2]
        # ========================== #

        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        """     Initialize Soft_indices     """
        soft_indices = []
        scale = [80, 40, 20]
        """================================="""
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices


        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):            # for loop for num of scale
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # gain = scale: [1, 1, 1, 1, 1, 1, 1] -> [1, 1, S, S, S, S, 1]

            # Match targets to anchors
            t = targets * gain            # [3, 417, 7] : anchors, targets, (image, class, x, y, w, h, anchor_idx)      # x, y, w, h를 scale 단위로 변환 targets: 0~1  --->  t: 0~scale 단위 
            if nt:
                # Matches / Matchin bbox with anchors
                # 해당 anchor랑 얼마나 비슷한지 게산 후 self.hyp['anchor_t'] 보다 작은 것들만 고른다.
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio [3, 417, 2]
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare [3, 417]
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # 해당 anchor와 어느정도 비슷한 predictions 만 남는다. shape : [417, 7] 7 -> image, class, x, y, w, h, anchor_idx

                # # Offsets / Neighboring cell assignment
                # gxy = t[:, 2:4]                                                 # grid xy shape: [417, 2]
                # gxi = gain[[2, 3]] - gxy                                        # inverse
                # j, k = ((gxy % 1 < g) & (gxy > 1)).T                            # j, k shape: [417], [417]
                # l, m = ((gxi % 1 < g) & (gxi > 1)).T
                # j = torch.stack((torch.ones_like(j), j, k, l, m))               # j는 mask. [기존 cell, lebft, upper, right, lower] 0 or 1 shape:[5, 417]
                # t = t.repeat((5, 1, 1))[j]                                      # [417, 7] ->  [5, 417, 7] -> masking(5개 중 최대 3개 선택) -> [1251, 7] 보통 각 cell 당 양옆위아래로 총 3개씩 배정
                # offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]       # [1, 417, 2] + [5, 1, 2] -> [5, 417, 2] -> masking -> [1251, 7]
            else:
                t = targets[0]
                # offsets = 0

            """     Shapes of Tensors           """
            """ t.shape = [num_targets, 7]      """
            """ gxy.shape = [num_targets, 2]    """
            """================================="""


            # Define
            b, c = t[:, :2].long().T                # image, class
            gxy = t[:, 2:4]                         # grid xy
            gwh = t[:, 4:6]                         # grid wh
            # gij = (gxy - offsets).long()          # grid cell index
            gij = (gxy).long()                      # grid cell index
            gi, gj = gij.T                          # grid x,y indices
            a = t[:, 6].long()                      # anchor indices
            # 나중에 gxy - gji 를 통해 cell 내에서의 좌표를 넘겨준다.
            
            



            "find the number of cells in each bouding boxes"
            gi1j1, gi2j2 = (gxy - gwh/2).long(), (gxy + gwh/2).long()           # top_left cell, bottom_rigtht cell index
            gi1, gj1  = gi1j1.T                                                 # top_left cell index
            width, height = (gi2j2 + 1 - gi1j1).T                               # width, height of num_of_cells in bbox
            num_of_cells = width * height                                       # number of cells in bbox
            # <========================= 2. 작은 박스로 overlap 되게
            sorted_idx_numcells = torch.argsort(num_of_cells, descending=True)  
            # <========================= 

            "make a tensor containing all cells that are in each bounding box"
            "by repeating center cell with number of cells in in each bouding box"
            "soft_t[6:] : bbox starting cell on the top-left corner"
            gw, gh = gwh.T
            soft_t = torch.cat((b.unsqueeze(-1), a.unsqueeze(-1), gj.unsqueeze(-1), gi.unsqueeze(-1), height.unsqueeze(-1), width.unsqueeze(-1), gj1.unsqueeze(-1), gi1.unsqueeze(-1)), dim=1)         # gj1, gi1 : top left cell index
            # <====== 2nd trial
            soft_t = soft_t[sorted_idx_numcells]                                           
            num_of_cells = num_of_cells[sorted_idx_numcells]                               
            width, height = width[sorted_idx_numcells], height[sorted_idx_numcells]         
            # <======================== 2
            soft_t = torch.repeat_interleave(soft_t, num_of_cells, dim=0)                                               # repeat every bbox with the number of cells in each bbox
            # pdb.set_trace()
            
            "reassign correct indices for each cell"
            start_idx = 0
            cell_offset = torch.tensor([], device = targets.device)
            for i in range(len(num_of_cells)):          # enumerate보다 이게 빠를려나 싶어서
                end_idx = num_of_cells[i] + start_idx
                w, h = width[i], height[i]
                soft_gi, soft_gj = torch.arange(w).to('cuda').repeat_interleave(h), torch.arange(h).to('cuda').repeat(w)
                soft_ji = torch.cat((soft_gj.unsqueeze(-1), soft_gi.unsqueeze(-1)), dim=1)          # indices for each cell in a bbox
                cell_offset = torch.cat((cell_offset, soft_ji), dim=0)
                # soft_t[start_idx:end_idx, 6:] += soft_ji 
                start_idx = end_idx
            soft_t[..., 6:] += cell_offset.long()

            
            """
            Assigning Normalized Distance and Score Assignment
            soft_t = [b, a, gj, gi, height, width, gj1, gi1]
            index:     0, 1, 2,  3,    4  ,   5  ,  6  , 7
            """
            S = gain[3]
            center_of_cells = soft_t[..., 6:] + 0.5
            normalized_distance = torch.sqrt(((soft_t[..., 2:3] - soft_t[..., 6:7])/(soft_t[..., 4:5]/2)) ** 2 + ((soft_t[..., 3:4] - soft_t[..., 7:])/(soft_t[...,5:6]/2)) ** 2)
            # normalized_distance = torch.sqrt(((soft_t[..., 2:3] - soft_t[..., 6:7])/(soft_t[..., 4:5])) ** 2 + ((soft_t[..., 3:4] - soft_t[..., 7:])/(soft_t[...,5:6])) ** 2)
            # score = (normalized_distance - 1) ** 30
            # score = (normalized_distance - 1) ** 10         # Best
            # score = (normalized_distance - 1) ** 4
            # score = (normalized_distance - 1) ** 20
            # score = (normalized_distance - 1) ** self.hyp["n"]
            # score = (normalized_distance - 1) ** (10 + 10 * (epoch//200))
            # score = (normalized_distance - 1) ** (10 + 10 * (epoch//100))

            """ score assignment """
            # score = (normalized_distance - 1) ** self.n                                   # score1
            score = (torch.exp(-( normalized_distance ** 2 / (2*( self.sigma**2)))))        # score2
            # print("sigma:",self.sigma)
            # print("score:",score)
            # pdb.set_trace()


            # pdb.set_trace()
            











            # Append for each scale
            # a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))    # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))                                     # box 해당 cell 에서 gt_point가 얼마나 떨어져있는지
            anch.append(anchors[a])                                                         # anchors
            tcls.append(c)                                                                  # class
            

            """     append soft mask and score      """
            # score = score.type(torch.double)                      
            # sorted_idx = torch.argsort(score, dim=0)                                #<=====         1st trial 기존의 큰값의 score로 overlap
            # sorted_idx = torch.argsort(score, dim=0, descending=True)              #<=====         2nd-1) trial 작은값의 score로 overlap 이렇게 하면 안됨!!!!!
            # score = score[sorted_idx]
            # soft_t = soft_t[sorted_idx]
            # pdb.set_trace() 

            b_soft, a_soft = soft_t[..., :2].T
            gj_soft, gi_soft = soft_t[..., 6:].T                 #<=====
            
            # # 3rd Trial
            # # 아래 처럼 변경: 겹치는 부분은 평균값을 부여
            # soft_gjgi, idx = torch.unique(soft_t[:, 6:], dim=0, return_inverse=True)
            # gj_soft, gi_soft = soft_gjgi.T
            # ba_soft = torch.zeros((len(soft_gjgi), 2), device = targets.device)
            # score_soft = torch.zeros(len(soft_gjgi), device = targets.device)
            # repeat = torch.zeros(len(soft_gjgi), device = targets.device)
            # for i, idxx in enumerate(idx):
            #     score_soft[idxx:idxx+1] += score[i]
            #     repeat[idxx:idxx+1] += 1
            #     ba_soft[idxx:idxx+1] = soft_t[i, 0:2]
            # b_soft, a_soft = ba_soft.T
            # score = score_soft / repeat




            soft_indices.append((b_soft.long(), a_soft.long(), gj_soft.long().clamp_(0, gain[3] - 1), gi_soft.long().clamp_(0, gain[2] - 1), score.view(1, -1).double()))
            # pdb.set_trace()
            """====================================="""

        return tcls, tbox, indices, anch, soft_indices



