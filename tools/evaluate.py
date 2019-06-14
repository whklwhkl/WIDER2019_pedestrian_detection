import numpy as np
import os
import argparse

def check_size(submission_file):
    max_size = 60*1024*1024
    if os.path.getsize(submission_file) > max_size:
        raise IOError("File size exceeds the specified maximum size, which is 60M for the server.")

def remove_ignored_det(dt_box, ig_box):
    remain_box = []
    for p in dt_box:
        if len(p)>4:
            _,pl,pt,pr,pb = p
        else:
            pl,pt,pr,pb = p
        p_area = float((pr-pl)*(pb-pt))
        overlap = -0.01
        for c in ig_box:
            cl,ct,cr,cb = c
            if (cr>pl) and (cl<pr) and (ct<pb) and (cb>pt):
                overlap += (min(cr,pr)-max(cl,pl)+1.0)*(min(cb,pb)-max(ct,pt)+1.0)
        if overlap/p_area <= 0.5:
            remain_box.append(p)
    return remain_box

def parse_ignore_file(ignore_file):
    with open(ignore_file, 'r') as f:
        lines = f.readlines()
    ignore = {}
    for line in lines:
        line = line.strip().split()
        image_id = line[0]
        bbox = []
        ignore_num = (len(line)-1)//4
        for i in range(ignore_num):
            b = []
            b.append(int(line[1+4*i]))
            b.append(int(line[2+4*i]))
            b.append(int(line[1+4*i])+int(line[3+4*i]))
            b.append(int(line[2+4*i])+int(line[4+4*i]))
            bbox.append(b)
        ignore[image_id] = bbox
    return ignore

def parse_gt_file(gt_file, ignore):
    with open(gt_file, 'r') as f:
        lines = f.readlines()
    gts = {}
    for line in lines:
        line = line.strip().split()
        image_id = line[0]
        BBOX = []
        bbox_num = (len(line)-1)//4
        for i in range(bbox_num):
            b = []
            b.append(int(line[1+4*i]))
            b.append(int(line[2+4*i]))
            b.append(int(line[3+4*i])+int(line[1+4*i]))
            b.append(int(line[2+4*i])+int(line[4+4*i]))
            BBOX.append(b)
        if image_id in ignore and len(BBOX):
            BBOX = remove_ignored_det(BBOX, ignore[image_id])
        gts[image_id] = {'bbox': BBOX}
    return gts

def parse_submission_file(sub_file, ignore, img_lst):
    with open(sub_file,'r') as f:
        lines = f.readlines()
    subs = {}
    for line in lines:
        line = line.strip().split()
        image_id = line[0]
        if image_id not in img_lst:
           raise KeyError("Can not find image {} in the groundtruth file, did you submit the result file for the right dataset?".format(image_id))
        subs.setdefault(image_id, [])
        subs[image_id].append([float(line[1]), float(line[2]), float(line[3]), float(line[4])+float(line[2]), float(line[5])+float(line[3])])
    for key in list(ignore.keys()):
        if key not in list(subs.keys()): continue
        subs[key] = remove_ignored_det(subs[key], ignore[key])

    final_subs = []
    for key in img_lst:
        if key not in list(subs.keys()): continue
        for item in subs[key]:
            final_subs.append({'image_id':key, 'score':item[0], 'bbox':item[1:]})
    final_subs = sorted(final_subs, key=lambda x: -x['score'])
    return final_subs

def compute_ap(rec, prec):
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def pedestrian_eval(dts, gt):
    aap = []
    nd = len(dts)
    ovethr = np.arange(0.5,1.0,0.05)
    for ove in ovethr:
        npos = 0
        for image_id in list(gt.keys()):
            npos += len(gt[image_id]['bbox'])
            gt[image_id]['det'] = [False] * len(gt[image_id]['bbox'])

        tp = np.zeros(nd)
        fp = np.zeros(nd)

        for i in range(nd):
            bb = dts[i]['bbox']
            image_id = dts[i]['image_id']
            BBGT = np.array(gt[image_id]['bbox'])
            ovmax = -np.inf
            if BBGT.size > 0:
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) + (BBGT[:, 2] - BBGT[:, 0] + 1.) * \
                    (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
            if ovmax > ove:
                if not gt[image_id]['det'][jmax]:
                    tp[i] = 1.
                    gt[image_id]['det'][jmax] = 1.
                else:
                    fp[i] = 1.
            else:
                fp[i] = 1.
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = compute_ap(rec, prec)
        aap.append(ap)
    mAP = np.mean(aap)
    return mAP

if __name__ == '__main__':
    import sys

    gt_file = 'Annotations/val_bbox.txt'
    ignore_file = 'Annotations/val_ignore.txt'
    # submit_file = 'submission_example.txt'
    submit_file = sys.argv[1]

    check_size(submit_file)
    ignore_zones = parse_ignore_file(ignore_file)
    gt = parse_gt_file(gt_file, ignore_zones)
    dts = parse_submission_file(submit_file, ignore_zones, sorted(gt.keys()))
    mAP = pedestrian_eval(dts, gt)

    out = {'Average AP': mAP}
    print(out)
    #strings = ['{}: {}\n'.format(k, v) for k, v in out.items()]
    #open(os.path.join(output_dir, 'scores.txt'), 'w').writelines(strings)
