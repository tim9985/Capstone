"""
losses_ext.py — NWD (Normalized Wasserstein Distance) 를 ultralytics 학습에 끼워 넣는다 (2026-09-25)

  작은 박스는 몇 px 만 어긋나도 IoU 가 급락한다 → 학습 중 양성 할당·박스 손실에서 작은 사람이 불리하다.
  NWD 는 박스를 2차원 가우시안으로 보고 거리를 잰다 (Wang et al. 2021 arXiv 2110.13389 · ISPRS 2022)
      W2² = |c1 − c2|² + (|w1 − w2|² + |h1 − h2|²) / 4        NWD = exp(−√W2² / C)

  바꾸는 곳 (평가는 그대로 IoU 0.5 AP50)
    ① 양성 할당 TaskAlignedAssigner.iou_calculation : (1−α)·CIoU + α·NWD
    ② 박스 손실 BboxLoss.forward                     : (1−α)·(1−CIoU) + α·(1−NWD)
  단위: 둘 다 **픽셀** (할당은 원래 픽셀 · 손실은 격자 단위라 stride 를 곱해 픽셀로)
  ultralytics 원본 파일은 고치지 않는다 — 같은 프로세스 안에서 메서드만 바꾼다 (8.4.102 에서 확인)
"""
import torch


def nwd(b1, b2, C):
    """xyxy 박스 쌍 (…,4) → (…) NWD ∈ (0,1]"""
    c1 = (b1[..., :2] + b1[..., 2:4]) / 2
    c2 = (b2[..., :2] + b2[..., 2:4]) / 2
    wh1 = (b1[..., 2:4] - b1[..., :2]).clamp(min=1e-6)
    wh2 = (b2[..., 2:4] - b2[..., :2]).clamp(min=1e-6)
    w2 = ((c1 - c2) ** 2).sum(-1) + ((wh1 - wh2) ** 2).sum(-1) / 4
    return torch.exp(-torch.sqrt(w2.clamp(min=1e-7)) / C)


def enable_nwd(alpha=0.5, C=32.0):
    """alpha: NWD 비중 (0 이면 원래와 같다) · C: 거리 기준 px (사람 긴 변 중앙값 ~45 px 근처)"""
    from ultralytics.utils import loss as L
    from ultralytics.utils import tal as T

    if getattr(T.TaskAlignedAssigner, "_nwd_on", False):
        return
    orig_iou = T.TaskAlignedAssigner.iou_calculation
    orig_fwd = L.BboxLoss.forward

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        ciou = orig_iou(self, gt_bboxes, pd_bboxes)
        return (1 - alpha) * ciou + alpha * nwd(gt_bboxes, pd_bboxes, C).to(ciou.dtype)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores,
                target_scores_sum, fg_mask, imgsz, stride):
        loss_iou, loss_dfl = orig_fwd(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                                      target_scores, target_scores_sum, fg_mask, imgsz, stride)
        if fg_mask.sum() == 0:
            return loss_iou, loss_dfl
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        s = stride.unsqueeze(0).expand(pred_bboxes.shape[0], -1, -1)[fg_mask]      # (n,1) 격자 → px
        n = nwd(pred_bboxes[fg_mask] * s, target_bboxes[fg_mask] * s, C).unsqueeze(-1)
        loss_nwd = ((1.0 - n) * weight).sum() / target_scores_sum
        return (1 - alpha) * loss_iou + alpha * loss_nwd, loss_dfl

    T.TaskAlignedAssigner.iou_calculation = iou_calculation
    L.BboxLoss.forward = forward
    T.TaskAlignedAssigner._nwd_on = True
    print(f"  NWD 켬      : 할당·박스 손실 = (1−{alpha})·CIoU + {alpha}·NWD · C = {C} px")
