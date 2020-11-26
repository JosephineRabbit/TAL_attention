import sys
import numpy as np
import pandas as pd
import json
import os
from joblib import Parallel, delayed

from gtad_lib import opts


def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data


def get_infer_dict(opt):
    df = pd.read_csv(opt["video_info"])
    json_data = load_json(opt["video_anno"])
    database = json_data
    video_dict = {}
    for i in range(len(df)):
        video_name = df.video.values[i]
        video_info = database[video_name]
        video_new_info = {}
        video_new_info['duration_frame'] = video_info['duration_frame']
        video_new_info['duration_second'] = video_info['duration_second']
        video_new_info["feature_frame"] = video_info['feature_frame']
        video_subset = df.subset.values[i]
        video_new_info['annotations'] = video_info['annotations']
        if video_subset == 'validation':
            video_dict[video_name] = video_new_info
    return video_dict


def Soft_NMS(df, nms_threshold=1e-5, num_prop=100):
    '''
    From BSN code
    :param df:
    :param nms_threshold:
    :return:
    '''
    df = df.sort_values(by="score", ascending=False)

    tstart = list(df.xmin.values[:])
    tend = list(df.xmax.values[:])
    tscore = list(df.score.values[:])

    rstart = []
    rend = []
    rscore = []

    # # frost: I use a trick here, remove the detection
    # # which is longer than 300
    # for idx in range(0, len(tscore)):
    #     if tend[idx] - tstart[idx] >= 300:
    #         tscore[idx] = 0

    while len(tscore) > 1 and len(rscore) < num_prop and max(tscore) > 0:
        max_index = tscore.index(max(tscore))
        for idx in range(0, len(tscore)):
            if idx != max_index:
                tmp_iou = IOU(tstart[max_index], tend[max_index], tstart[idx], tend[idx])
                if tmp_iou > 0:
                    tscore[idx] = tscore[idx] * np.exp(-np.square(tmp_iou) / nms_threshold)

        rstart.append(tstart[max_index])
        rend.append(tend[max_index])
        rscore.append(tscore[max_index])
        tstart.pop(max_index)
        tend.pop(max_index)
        tscore.pop(max_index)

    newDf = pd.DataFrame()
    newDf['score'] = rscore
    newDf['xmin'] = rstart
    newDf['xmax'] = rend
    return newDf


def IOU(s1, e1, s2, e2):
    if (s2 > e1) or (s1 > e2):
        return 0
    Aor = max(e1, e2) - min(s1, s2)
    Aand = min(e1, e2) - max(s1, s2)
    return float(Aand) / Aor



def gen_detection_multicore(opt):
    # get video duration
    infer_dict = get_infer_dict(opt)

    # load class name and video level classification
    cls_data = load_json("data/cuhk_val_simp_share.json")
    # cls_data_score, cls_data_action = cls_data["results"], cls_data["class"]
    cls_data_score, cls_data_cls = {}, {}
    for idx, vid in enumerate(infer_dict.keys()):
        vid = vid[2:]
        cls_data_score[vid] = np.array(cls_data["results"][vid])
        cls_data_cls[vid] = cls_data["class"]  # [np.argmax(cls_data_score[vid])] # find the max class




if __name__ == "__main__":
    opt = opts.parse_opt()
    opt = vars(opt)

   # print("Detection post processing start")
   # gen_detection_multicore(opt)
   # print("Detection Post processing finished")

    from evaluation.eval_detection import ANETdetection

    anet_detection = ANETdetection(
        ground_truth_filename="./evaluation/activity_net_1_3_new.json",
        prediction_filename=os.path.join(opt['output'], "detection_result_nms{}.json".format(opt['nms_thr'])),
        subset='validation', verbose=False, check_status=False)

    anet_detection.evaluate()

    mAP_at_tIoU = [f'mAP@{t:.2f} {mAP * 100:.3f}' for t, mAP in zip(anet_detection.tiou_thresholds, anet_detection.mAP)]
    results = f'Detection: average-mAP {anet_detection.average_mAP * 100:.3f} {" ".join(mAP_at_tIoU)}'
    print(results)
    with open(os.path.join(opt['output'], 'results.txt'), 'a') as fobj:
        fobj.write(f'{results}\n')


