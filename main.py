import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import tiffslide
import torch
import numpy as np
import cv2
from network.main_net11_me_npsamp import MainNet
from pred_Utils import run_pred_img
import json


SPACING_LEVEL0 = 0.24199951445730394

def px_to_mm(px: int, spacing: float):
    return px * spacing / 1000

def write_json_file(location, content):
    with open(location, 'w') as f:
        f.write(json.dumps(content, indent=4))


def get_model(ck_best_path,cla_class_num):
    b1_out_dim = 1
    b2_out_dim = 1
    b3_out_dim = cla_class_num
    net = MainNet(3, b1_out_dim, b2_out_dim, b3_out_dim)
    net.enabled_b2_branch = True
    net.enabled_b3_branch = True
    net.load_state_dict(torch.load(ck_best_path, 'cpu'))
    net = net.cuda()
    net.eval()
    return net


def run(net,wsi_tif_path,tissue_mask_path,name,out_dir):
    # name = os.path.basename(wsi_tif_path).replace('_PAS_CPG.tif','')

    img_slide = tiffslide.TiffSlide(wsi_tif_path)

    level = 3
    mask_slide = tiffslide.TiffSlide(tissue_mask_path)
    mask = mask_slide.read_region((0,0),level,mask_slide.level_dimensions[level],as_array=True)[:,:,0]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_ds = mask_slide.level_downsamples[level]

    lymphocytes_list = []
    monocytes_list = []
    inflammatory_cells_list = []

    list000 = []

    for cnt in contours:
        cnt = cnt[:,0,:]
        x_min, x_max, y_min, y_max = round(np.min(cnt[:, 0])*mask_ds), round(np.max(cnt[:, 0])*mask_ds), round(np.min(cnt[:, 1])*mask_ds), round(np.max(cnt[:, 1])*mask_ds)
        set_boundary = 128
        x_min, x_max, y_min, y_max  = x_min-set_boundary, x_max+set_boundary, y_min-set_boundary, y_max+set_boundary
        mask_tmp  = mask_slide.read_region((x_min, y_min), 0, (x_max - x_min, y_max - y_min), as_array=True)[:, :, 0]
        img_tmp   = img_slide.read_region((x_min, y_min), 0, (x_max - x_min, y_max - y_min), as_array=True)[:, :, :3]
        img_tmp[mask_tmp == 0] = [255, 255, 255]
        out_cla_pts_Hy_Wx,out_cla_pts_cla,out_cla_pts_cla_prob = run_pred_img(net,img_tmp,cla_class_num = cla_class_num,net_in_hw = net_in_hw)
        del img_tmp,mask_tmp

        for pts_yx,pts_cla,pts_cla_prob_3 in zip(out_cla_pts_Hy_Wx,out_cla_pts_cla,out_cla_pts_cla_prob):
            pts_cla_prob = float(pts_cla_prob_3[pts_cla])
            pts_cla_prob_other = float(pts_cla_prob_3[2])
            point_x,point_y = round(x_min+pts_yx[1]), round(y_min+pts_yx[0])
            point_x = px_to_mm(point_x, SPACING_LEVEL0)
            point_y = px_to_mm(point_y, SPACING_LEVEL0)
            list000.append((point_x,point_y,pts_cla,pts_cla_prob_3[0],pts_cla_prob_3[1],pts_cla_prob_3[2]))
            if   pts_cla == 0:
                dict_tmp = {"name": "Point " + str(len(lymphocytes_list) + 1), "point": [point_x,point_y, 0.24],"probability": pts_cla_prob}
                dict_tmp1 = {"name": "Point " + str(len(inflammatory_cells_list) + 1), "point": [point_x,point_y, 0.24],"probability": 1-pts_cla_prob_other}
                lymphocytes_list.append(dict_tmp)
                inflammatory_cells_list.append(dict_tmp1)
            elif pts_cla == 1:
                dict_tmp = {"name": "Point " + str(len(monocytes_list) + 1), "point": [point_x,point_y, 0.24],"probability": pts_cla_prob}
                dict_tmp1 = {"name": "Point " + str(len(inflammatory_cells_list) + 1), "point": [point_x,point_y, 0.24],"probability": 1-pts_cla_prob_other}
                monocytes_list.append(dict_tmp)
                inflammatory_cells_list.append(dict_tmp1)
            else:
                pass
    os.makedirs(out_dir+'/'+name,exist_ok=True)
    pred_lymphocytes = {
        "name": "lymphocytes",
        "type": "Multiple points",
        "points": lymphocytes_list,
        "version": {"major": 1, "minor": 0}
    }
    pred_monocytes = {
        "name": "monocytes",
        "type": "Multiple points",
        "points": monocytes_list,
        "version": {"major": 1, "minor": 0}
    }
    pred_inflammatory_cells = {
        "name": "inflammatory_cells",
        "type": "Multiple points",
        "points": inflammatory_cells_list,
        "version": {"major": 1, "minor": 0}
    }

    json_filename_lymphocytes = out_dir+'/'+ "detected-lymphocytes.json"
    json_filename_monocytes = out_dir+'/'+"detected-monocytes.json"
    json_filename_inflammatory_cells = out_dir+'/'+ "detected-inflammatory-cells.json"
    write_json_file(json_filename_lymphocytes,pred_lymphocytes)
    write_json_file(json_filename_monocytes,pred_monocytes)
    write_json_file(json_filename_inflammatory_cells,pred_inflammatory_cells)

if __name__ == '__main__':

    from pathlib import Path
    from glob import glob
    model_path = './115_2.pth'
    cla_class_num = 3
    net_in_hw = (128, 128)
    net = get_model(model_path, cla_class_num)

    INPUT_PATH = Path("/input")
    out_dir = "/output"

    image_paths = sorted(glob(os.path.join(INPUT_PATH, "images/kidney-transplant-biopsy-wsi-pas/*.tif")))
    mask_paths = sorted(glob(os.path.join(INPUT_PATH, "images/tissue-mask/*.tif")))


    list_all = []
    for wsi_tif_path,tissue_mask_path in zip(image_paths,mask_paths):
        name = os.path.basename(wsi_tif_path).replace('_PAS_CPG.tif','')

        run(net,str(wsi_tif_path),str(tissue_mask_path),name,out_dir)

        dict_tmp = {
            "pk":  name,
            "inputs": [
                {
                    "image": {
                        "name": name+"_PAS_CPG.tif"
                    },
                    "interface": {
                        "slug": "kidney-transplant-biopsy",
                        "kind": "Image",
                        "super_kind": "Image",
                        "relative_path": "images/kidney-transplant-biopsy-wsi-pas"
                    }
                }
            ],
            "outputs": [
                {
                    "interface": {
                        "slug": "detected-lymphocytes",
                        "kind": "Multiple points",
                        "super_kind": "File",
                        "relative_path": "detected-lymphocytes.json"
                    }
                },
                {
                    "interface": {
                        "slug": "detected-monocytes",
                        "kind": "Multiple points",
                        "super_kind": "File",
                        "relative_path": "detected-monocytes.json"
                    }
                },
                {
                    "interface": {
                        "slug": "detected-inflammatory-cells",
                        "kind": "Multiple points",
                        "super_kind": "File",
                        "relative_path": "detected-inflammatory-cells.json"
                    }
                }
            ]
        }

        list_all.append(dict_tmp)


    with open(out_dir+'/predictions.json', 'w') as f:
        f.write(json.dumps(list_all, indent=4))
