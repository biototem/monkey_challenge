import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import tiffslide
import torch
import numpy as np
import cv2
from network.main_net11_me_npsamp import MainNet
from pred_Utils import run_pred_img
import json
from dot2polygon import dot2polygon
import xml.etree.ElementTree as ET
SPACING_LEVEL0 = 0.24199951445730394
import torch


def match_coordinates(ground_truth, predictions, pred_prob, margin):
    """
    Matches predicted coordinates to ground truth coordinates within a certain distance margin
    and computes the associated probabilities for true positives and false positives.

    Args:
        ground_truth (list of tuples): List of ground truth coordinates as (x, y).
        predictions (list of tuples): List of predicted coordinates as (x, y).
        pred_prob (list of floats): List of probabilities associated with each predicted coordinate.
        margin (float): The maximum distance for considering a prediction as a true positive.

    Returns:
        true_positives (int): Number of correctly matched predictions.
        false_negatives (int): Number of ground truth coordinates not matched by any prediction.
        false_positives (int): Number of predicted coordinates not matched by any ground truth.
        tp_probs (list of floats): Probabilities of the true positive predictions.
        fp_probs (list of floats): Probabilities of the false positive predictions.
    """
    if len(ground_truth) == 0 and len(predictions) == 0:
        return 0, 0, 0, np.array([]), np.array([])
        # return true_positives, false_negatives, false_positives, np.array(tp_probs), np.array(fp_probs)
    # Convert lists to numpy arrays for easier distance calculations
    gt_array = np.array(ground_truth)
    pred_array = np.array(predictions)
    pred_prob_array = np.array(pred_prob)

    # 将数组转换为 PyTorch 张量，并移动到 GPU
    gt_tensor = torch.tensor(gt_array, device='cuda',dtype=torch.float64)
    pred_tensor = torch.tensor(pred_array, device='cuda',dtype=torch.float64)

    # 计算欧几里得距离矩阵
    dist_matrix_gpu = torch.cdist(gt_tensor, pred_tensor)
    # print(11111111111111)
    # 如果需要将结果转回 CPU 上

    # print(999,np.allclose(dist_matrix,dist_matrix1,rtol = 0.01,atol = 0.01))
    # Initialize sets for matched indices
    matched_gt = set()
    matched_pred = set()

    # Iterate over the distance matrix to find the closest matches
    while True:
        # Find the minimum distance across all av min_dist = np.min(dist_matrix)
        min_dist = torch.min(dist_matrix_gpu)

        # If the minimum distance exceeds the margin or no valid pairs left, break
        if min_dist > margin or min_dist == np.inf:
            break

        # Get the indices of the GT and prediction points with the minimum distance
        gt_idx, closest_pred_idx = torch.unravel_index(torch.argmin(dist_matrix_gpu), dist_matrix_gpu.shape)
        gt_idx, closest_pred_idx = int(gt_idx), int(closest_pred_idx)
        # Mark these points as matched
        matched_gt.add(gt_idx)
        matched_pred.add(closest_pred_idx)

        # Set the row and column of the matched points to infinity in the original distance matrix
        dist_matrix_gpu[gt_idx, :] = np.inf  # Mask this GT point
        dist_matrix_gpu[:, closest_pred_idx] = np.inf  # Mask this Pred point

    # Calculate true positives, false negatives, and false positives
    true_positives = len(matched_gt)
    false_negatives = len(ground_truth) - true_positives
    false_positives = len(predictions) - true_positives
    TP = true_positives
    FP = false_positives
    FN = false_negatives
    # 计算准确率
    precision = TP / (TP + FP)
    # 计算召回率
    recall = TP / (TP + FN)
    # 计算 f1 分数

    F1_Score = 2 * (precision * recall) / (precision + recall)


    # Compute probabilities for true positives and false positives
    tp_probs = [pred_prob[i] for i in matched_pred]
    fp_probs = [pred_prob[i] for i in range(len(predictions)) if i not in matched_pred]

    return true_positives, false_negatives, false_positives, np.array(tp_probs), np.array(fp_probs),F1_Score

from monai.metrics import compute_froc_score, compute_froc_curve_data
from scipy.spatial import distance
from sklearn.metrics import auc

def get_froc_vals(gt_dict, result_dict, radius: int):
    """
    Computes the Free-Response Receiver Operating Characteristic (FROC) values for given ground truth and result data.
    Using https://docs.monai.io/en/0.5.0/_modules/monai/metrics/froc.html
    Args:
        gt_dict (dict): Ground truth data containing points and regions of interest (ROIs).
        result_dict (dict): Result data containing detected points and their probabilities.
        radius (int): The maximum distance in pixels for considering a detection as a true positive.

    Returns:
        dict: A dictionary containing FROC metrics such as sensitivity, false positives per mm²,
              true positive probabilities, false positive probabilities, total positives,
              area in mm², and FROC score.
    """
    # in case there are no predictions
    if len(result_dict['points']) == 0:
        return {'sensitivity_slide': [0], 'fp_per_mm2_slide': [0], 'fp_probs_slide': [0],
                'tp_probs_slide': [0], 'total_pos_slide': 0, 'area_mm2_slide': 0, 'froc_score_slide': 0}
    if len(gt_dict['points']) == 0:
        return None

    gt_coords = [i['point'] for i in gt_dict['points']]
    gt_rois = [i['polygon'] for i in gt_dict['rois']]
    # compute the area of the polygon in roi
    area_mm2 = SPACING_LEVEL0 * SPACING_LEVEL0 * gt_dict["area_rois"] / 1000000
    # result_prob = [i['probability'] for i in result_dict['points']]
    result_prob = [i['probability'] for i in result_dict['points']]


    # a331 = np.array(result_prob)
    # print(np.sum(a331<0.55)/a331.shape[0])

    result_coords = [[i['point'][0], i['point'][1]] for i in result_dict['points']]

    # prepare the data for the FROC curve computation with monai
    true_positives, false_negatives, false_positives, tp_probs, fp_probs,F1_Score = match_coordinates(gt_coords, result_coords,  result_prob, radius)
    TP, FN, FP = true_positives,false_negatives,false_positives

    # 计算 Precision 和 Recall
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    # 计算 F1 Score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    # print(f1, recall)

    total_pos = len(gt_coords)
    # the metric is implemented to normalize by the number of images, we however want to have it by mm2, so we set
    # num_images = ROI area in mm2
    fp_per_mm2_slide, sensitivity = compute_froc_curve_data(fp_probs, tp_probs, total_pos, area_mm2)
    if len(fp_per_mm2_slide) > 1 and len(sensitivity) > 1:
        area_under_froc = auc(fp_per_mm2_slide, sensitivity)
        froc_score = compute_froc_score(fp_per_mm2_slide, sensitivity, eval_thresholds=(10, 20, 50, 100, 200, 300))
    else:
        area_under_froc = 0
        froc_score = 0

    return {'froc_score_slide': float(froc_score), 'F1_score': f1,'rec':recall}
def draw_cla_im(im, pts, cls):
    cls_color = {
            0: (205, 0,   0),#Neoplastic
            1: (238,238,0),#Inflammatory
        }
    assert len(pts) == len(cls)
    im = im.copy()
    for pt, c in zip(pts, cls):
        # pt = pt * 2
        cv2.circle(im, tuple(pt)[::-1], 3, cls_color[c], 3)
    return im


def px_to_mm(px: int, spacing: float):
    return px * spacing / 1000

def write_json_file(location, content):
    with open(location, 'w') as f:
        f.write(json.dumps(content, indent=4))


def get_model(ck_best_path,cla_class_num):
    b1_out_dim = 1
    b2_out_dim = 1
    b3_out_dim = 3
    net = MainNet(3, b1_out_dim, b2_out_dim, b3_out_dim)
    net.enabled_b2_branch = True
    net.enabled_b3_branch = True
    net.load_state_dict(torch.load(ck_best_path, 'cpu'))
    net = net.cuda()
    net.eval()
    return net

def dot2polygon_cell1(xml_path ):

    # 解析XML文件
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 创建两个字典来存储结果
    xy_list = []
    xy_list_type = []

    # 遍历所有Annotation元素
    for annotation in root.findall(".//Annotation"):
        # 检查Annotation的标签
        part_of_group = annotation.get("PartOfGroup")
        # 查找特定标签的坐标
        if part_of_group == "Lymphocytes" or part_of_group == "Monocytes":
            for coordinate in annotation.findall(".//Coordinate"):
                x = float(coordinate.get("X"))
                y = float(coordinate.get("Y"))
                # 将结果添加到相应的列表
                if part_of_group == "Lymphocytes":
                    xy_list.append([x, y])
                    xy_list_type.append(0)
                elif part_of_group == "Monocytes":

                    xy_list.append([x, y])
                    xy_list_type.append(1)
    xy_list1 = np.array(xy_list).astype(np.int32)
    xy_list22 = np.array(xy_list_type).astype(np.int32)
    return xy_list1,xy_list22
def dot2polygon_cell2(xml_path ):
    '''
    :param xml_path (str): the path of the annotation file, ex. root\sub_root\filename.xml
    :param lymphocyte_half_box_size (folat): the size of half of the bbox around the lymphocyte dot in um, 4.5 for lymphocyte
    :param monocytes_half_box_size (folat): the size of half of the bbox around the monocytes dot in um, 11.0 for monocytes
    :param min_spacing (float): the minimum spacing of the wsi corresponding to the annotations
    :param output_path (str): the output path
    :return:
    '''


    # parsing the annotation
    tree = ET.parse(xml_path)
    root = tree.getroot()

    xy_list = []
    xy_list_type = []
    # iterating through the dot annotation.
    for A in root.iter('Annotation'):
        #Lymphocytes:
        if (A.get('PartOfGroup')=="lymphocytes") & (A.get('Type')=="Dot"):
        # change the type to Polygon
            A.attrib['Type'] = "Polygon"

            for child in A:
                for sub_child in child:
                    x_value = float(sub_child.attrib['X'])
                    y_value = float(sub_child.attrib['Y'])
                    xy_list.append([x_value,y_value])
                    xy_list_type.append(0)
        # Monoocytes:
        if (A.get('PartOfGroup')=="monocytes") & (A.get('Type')=="Dot"):
        # change the type to Polygon
            A.attrib['Type'] = "Polygon"
            for child in A:
                for sub_child in child:
                    x_value = float(sub_child.attrib['X'])
                    y_value = float(sub_child.attrib['Y'])
                    xy_list.append([x_value, y_value])
                    xy_list_type.append(1)
    xy_list1 = np.array(xy_list).astype(np.int32)
    xy_list22 = np.array(xy_list_type).astype(np.int32)
    return xy_list1,xy_list22
import imageio
def load_json_file(location):
    # Reads a json file
    with open(location) as f:
        return json.loads(f.read())


model_path ='/home/he/桌面/data/qietu/tmp/model/115_2.pth'
cla_class_num = 3
net_in_hw = (128,128)
net = get_model(model_path,cla_class_num)
val_list =         ["A_P000017",
    "D_P000002",
    "A_P000029",
    "C_P000027",
    "A_P000003",
    "C_P000024",
    "B_P000017",
    "A_P000018",
    "C_P000034",
    "B_P000001",
    "C_P000025",
    "D_P000015",
    "A_P000034",
    "A_P000035",
    "C_P000040",
    "A_P000016",
    "D_P000004",
    "C_P000030",
    "A_P000022"]
wsi_dir  = '/home/he/桌面/data/images/pas-cpg/'
xml_dir = '/home/he/桌面/data/annotations/xml/'
out_dir = '/home/he/桌面/code/instanseg-main/instanseg/models/12_22/'
STD = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))
MEAN = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))



for iii0000 in [0.3,0.4,0.5,0.55,0.6,0.7]:
    for iii11111 in [0.3, 0.4, 0.5, 0.55, 0.6, 0.7]:
        list44444444 = []
        for ii333 in val_list:
            ii3331 = ii333+'_PAS_CPG.tif'
            wsi_tif_path = wsi_dir+ii3331
            name = ii3331.replace('_PAS_CPG.tif','')
            img_slide = tiffslide.TiffSlide(wsi_tif_path)

            in_xml = xml_dir + ii3331.replace('_PAS_CPG.tif', '.xml')

            list_cell1,list_cell2 = dot2polygon_cell1(in_xml)
            if len(list_cell1)==0:
                list_cell1,list_cell2 = dot2polygon_cell2(in_xml)

            cnt_list = dot2polygon(in_xml)
            colour_list = [(205,0,0),(238,238,0)]

            lymphocytes_list = []
            monocytes_list = []
            inflammatory_cells_list = []
            aaa1=0
            list000 = []
            wsi_w,wsi_h = img_slide.level_dimensions[0]
            for idx999,cnt in enumerate(cnt_list):
                x_min,x_max,y_min,y_max = np.min(cnt[:,0]),np.max(cnt[:,0]),np.min(cnt[:,1]),np.max(cnt[:,1])
                set_boundary = 64
                x_min, x_max, y_min, y_max  = x_min-set_boundary, x_max+set_boundary, y_min-set_boundary, y_max+set_boundary
                img_tmp   = img_slide.read_region((x_min, y_min), 0, (x_max - x_min, y_max - y_min), as_array=True)[:, :, :3]
                img_tmp_vv = img_tmp.copy()
                cnt1111 = cnt.copy()
                cnt1111[:,0]  =cnt1111[:,0]-x_min
                cnt1111[:,1]  =cnt1111[:,1]-y_min
                cv2.drawContours(img_tmp_vv, [cnt1111], 0, (0,0,0),5)


                mask = np.zeros((wsi_h,wsi_w), dtype=np.uint8)
                cv2.drawContours(mask, [cnt], 0, 1, cv2.FILLED)
                mask = mask[y_min:y_max,x_min:x_max]
                img_tmp[mask == 0] = [255, 255, 255]
                out_cla_pts_Hy_Wx,out_cla_pts_cla,out_cla_pts_cla_prob = run_pred_img(net,img_tmp.copy(),cla_class_num = cla_class_num,net_in_hw = net_in_hw,iii00=iii0000,iii11=iii11111)

                for pts_yx,pts_cla,pts_cla_prob_3 in zip(out_cla_pts_Hy_Wx,out_cla_pts_cla,out_cla_pts_cla_prob):
                    pts_cla_prob = float(pts_cla_prob_3[pts_cla])
                    pts_cla_prob_other = float(pts_cla_prob_3[2])

                    if pts_cla!=2:
                        cv2.circle(img_tmp_vv, (pts_yx[1], pts_yx[0]), 9 ,colour_list[pts_cla] , 2 )

                    point_x,point_y = int(x_min+pts_yx[1]), int(y_min+pts_yx[0])
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
                for cell_xy,type22 in zip(list_cell1,list_cell2):
                    cv2.circle(img_tmp_vv, (cell_xy[0]-x_min, cell_xy[1]-y_min), 3 ,colour_list[type22] , cv2.FILLED )
                imageio.v3.imwrite('/home/he/桌面/data/12_22/' + name + '#' + str(idx999) + '.png', img_tmp_vv)
            os.makedirs(out_dir + name, exist_ok=True)

            json_filename_lymphocytes = out_dir + name + '/' + "detected-lymphocytes.json"
            json_filename_monocytes = out_dir + name + '/' + "detected-monocytes.json"
            json_filename_inflammatory_cells = out_dir + name + '/' + "detected-inflammatory-cells.json"
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

            write_json_file(json_filename_lymphocytes, pred_lymphocytes)
            write_json_file(json_filename_monocytes, pred_monocytes)
            write_json_file(json_filename_inflammatory_cells, pred_inflammatory_cells)

            gt_inf_cells = load_json_file(r'/home/he/桌面/data/annotations/json/' + name + '_inflammatory-cells.json')
            pred_inf_cells = load_json_file(out_dir + name + '/' + "detected-inflammatory-cells.json")
            inflamm_froc = get_froc_vals(gt_inf_cells, result_dict=pred_inf_cells, radius=int(
                5 / SPACING_LEVEL0))  # margin for inflammatory cells is 7.5um at spacing 0.24 um / pixel

            # print('froc', inflamm_froc['froc_score_slide'], 'F1_score', inflamm_froc['F1_score'], 'recall', inflamm_froc['rec'])
            list44444444.append(inflamm_froc['froc_score_slide'])

            os.makedirs(out_dir+name,exist_ok=True)
        print(iii0000,iii11111,np.mean(list44444444))