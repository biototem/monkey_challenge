import torch
torch.set_grad_enabled(False)
import numpy as np
import utils.eval_utils as eval_utils
from utils.big_pic_result import BigPicPatch
from utils.heatmap_nms import heatmap_nms
def run_pred_img(net,im,cla_class_num,net_in_hw = (128,128),batch_size = 8,device = 'cuda:0'):
    use_heatmap_nms = True
    wim = BigPicPatch(1+1+cla_class_num, [im], (0, 0), window_hw=net_in_hw, level_0_patch_hw=(1, 1), custom_patch_merge_pipe=eval_utils.patch_merge_func, patch_border_pad_value=255, ignore_patch_near_border_ratio=0.5)
    gen = wim.batch_get_im_patch_gen(batch_size * 3)
    for batch_info, batch_patch0 in gen:
        # batch_patch = torch.tensor(np.asarray(batch_patch), dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255.
        batch_patch1  =[]
        STD = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))
        MEAN = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
        for i1u3 in batch_patch0:
            im111 = ((i1u3 / 255 - MEAN) / STD).transpose((2, 0, 1)).astype(np.float32)
            batch_patch1.append(im111)
        batch_patch = torch.tensor(np.asarray(batch_patch1), dtype=torch.float32, device=device)

        with torch.no_grad():
            batch_pred_det, batch_pred_det2, batch_pred_cla = net(batch_patch)
            batch_pred_det = batch_pred_det.clamp(0, 1)
            batch_pred_det2 = batch_pred_det2.clamp(0, 1)
            batch_pred_cla = batch_pred_cla.clamp(0, 1)
            batch_pred = torch.cat([batch_pred_det, batch_pred_det2, batch_pred_cla], 1)
            batch_pred = batch_pred.permute(0, 2, 3, 1).cpu().numpy()
        wim.batch_update_result(batch_info, batch_pred)

    pred_pm = wim.multi_scale_result[0].data / np.clip(wim.multi_scale_mask[0].data, 1e-8, None)
    pred_det_rough_pm, pred_det_fine_pm, pred_cla_pm = np.split(pred_pm, [1, 2], -1)
    # pred_det_final_pm = pred_det_rough_pm * pred_det_fine_pm
    pred_det_final_pm =   np.uint8(pred_det_rough_pm>0.6) * pred_det_fine_pm
    pred_cla_final_pm = pred_det_final_pm * pred_cla_pm
    if use_heatmap_nms:
        pred_det_final_pm[..., 0] = pred_det_final_pm[..., 0] * heatmap_nms(pred_det_final_pm[..., 0])

    pred_det_post_pts = eval_utils.get_pts_from_hm(pred_det_final_pm, 0.5)
    pred_cla_final_pm_hm_before = np.copy(pred_cla_final_pm)
    if use_heatmap_nms:
        for c in range(pred_cla_final_pm.shape[2]):
            pred_cla_final_pm[:, :, c] = pred_cla_final_pm[:, :, c] * heatmap_nms(pred_cla_final_pm[:, :, c])
    pred_cla_pts = []
    pred_cla_pts_cla = []
    pred_cla_pts_cla_prob = []
    pred_cla_pts.extend(pred_det_post_pts)
    cla_list1 , cla_list2 = eval_utils.get_cls_pts_from_hm_bingdou(pred_cla_pts, pred_cla_final_pm_hm_before)
    pred_cla_pts_cla.extend(cla_list1)
    pred_cla_pts_cla_prob.extend(cla_list2)
    return pred_cla_pts,pred_cla_pts_cla,pred_cla_pts_cla_prob