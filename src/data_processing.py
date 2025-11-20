#.py file to store the functions for processing
from collections import defaultdict
import re
import os
from multiprocessing import Pool
import multiprocessing
from pathlib import Path
import pickle as pkl
import argparse
import yaml

import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

#function used to ...
def normalize_image(image, perc):
    # Find the 5th and 95th percentile intensity values
    lower = np.percentile(image, perc[0])
    upper = np.percentile(image, perc[1])

    # Clip and normalize the image
    image = np.clip(image, lower, upper)
    normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

    return normalized_image.astype(np.uint8)

#function to draw classified labels on raw images
def draw_circles(circles, image, labels=None):
    lab_colors = {
        'Agg': (255,0,0),
        'Hom': (0,0,255),
        'Con': (0,255,0)  
    }
    
    
    image = image.copy()
    image = np.dstack([image, image, image])

    if labels is None:
        labels = [None for _ in circles]

    for i, lab in zip(circles, labels):
        center = (int(i[0][0]), int(i[0][1]))  # Circle center
        radius = int(i[1])  # Circle radius

        cv2.circle(image, center, 1, (0, 100, 100), 3)  # Draw circle center
        
        if lab is not None:
            cv2.putText(image, lab, center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.circle(image, center, radius, lab_colors[lab.split(' ')[0]], 2)  # Draw circle perimeter
        else:
            cv2.circle(image, center, radius, (255, 0, 255), 2)  # Draw circle perimeter


    return image

#function to search for specific directory names
def get_dirs_matching_regex(directory_path, pattern):
    for root, dirs, files in os.walk(directory_path):
        for dname in dirs:
            m = re.search(pattern, dname)
            if m is None:
                continue
            yield root, dname

#function to search for specific file names
def get_files_matching_regex(directory_path, pattern):
    for root, dirs, files in os.walk(directory_path):
        for fname in files:
            m = re.search(pattern, fname)
            if m is None:
                continue
            yield root, fname
            
#use 4-quandrant images and split those in seperate quadrant tifs
def extract_quadrants(in_pth, quadrant_map):
    quadrant_dirs = set()

    skipped = []
    for cur_exp_pth, (classify_quad_id, bbox_quad_id) in tqdm(quadrant_map.items()):
        #cur_pth = f"{in_pth}/{cur_exp_pth}"
        cur_pth = in_pth
        print(cur_pth)
        
        #will find .tifs named 1-1.tif, 1_1.tif, 1.tif, 1_1A.tif, 1-1A.tif in keep only those in data root
        main_tifs = list(sorted(get_files_matching_regex(cur_pth, "^[1-9]+(?:[-_]?[1-9]+[A-Za-z]?)?\.tif$")))
        main_tifs = [main_tif for main_tif in main_tifs if main_tif[0]==cur_pth]
        print(f"Following quandrant .tifs have been found {main_tifs}")
        
        #remove those that start with MAC based prefix
        main_tifs = [main_tif for main_tif in main_tifs if not main_tif[1].startswith("._")]

        for cur_dir, cur_fname in main_tifs:
            m = re.search("(?m)^(\d+).*\.tif$", cur_fname)
            im_idx = int(m.groups()[0])

            with open(f"{cur_dir}/ca_params.yml", "r") as fp:
                parms = yaml.safe_load(fp)
            lo = parms["quadrant_align_info"]["qpos"][classify_quad_id][:2]
            hi = parms["quadrant_align_info"]["qpos"][classify_quad_id][2:]

            if parms["quadrant_align_info"]["ref"] == classify_quad_id:
                M = None
            else:
                M = np.array(
                    parms["quadrant_align_info"]["warp"][str(classify_quad_id)]
                )

            quadrant_dirs.add((f"{cur_dir}/quadrant_{classify_quad_id}/", bbox_quad_id))
            Path(f"{cur_dir}/quadrant_{classify_quad_id}/raw/").mkdir(
                parents=True, exist_ok=True
            )

            im = Image.open(f"{cur_dir}/{cur_fname}")
            shape = (im.height, im.width)
            lo = tuple(round(xx * yy) for xx, yy in zip(lo, shape))
            hi = tuple(round(xx * yy) for xx, yy in zip(hi, shape))

            for i in tqdm(range(im.n_frames)):
                cur_out_pth = (
                    f"{cur_dir}/quadrant_{classify_quad_id}/raw/{im_idx}_{i}.tif"
                )
                if Path(cur_out_pth).exists():
                    skipped.append(cur_fname)
                    break

                im.seek(i)

                cur_quadrant = np.array(im)[lo[1] : hi[1], lo[0] : hi[0]]
                if M is None:
                    # Identitiy transform; this is the reference quadrant
                    pass
                else:
                    cur_quadrant = cv2.warpAffine(cur_quadrant, M, cur_quadrant.shape)
                Image.fromarray(cur_quadrant).save(cur_out_pth)

    skipped_str = "\n".join(skipped)
    print(
        f"Skipped the following quadrants that were already extracted:\n{skipped_str}"
    )
    return quadrant_dirs

def pad_arrays_to_max_size(arr_list):
    # Determine the maximum size along each dimension
    max_shape = np.array([arr.shape for arr in arr_list]).max(axis=0)

    padded_arr_list = []
    for arr in arr_list:
        # Calculate padding sizes for each dimension
        pad_width = []
        for dim_size, max_dim_size in zip(arr.shape, max_shape):
            pad_size = max_dim_size - dim_size
            pad_before = pad_size // 2
            pad_after = pad_size - pad_before
            pad_width.append((pad_before, pad_after))

        # Apply padding
        arr_padded = np.pad(arr, pad_width, mode="constant", constant_values=0)
        padded_arr_list.append(arr_padded)

    return padded_arr_list

#write each droplet image for debugging purposes
def write_droplet_debug_ims(quadrant_dir):
    if Path(f"{quadrant_dir}/droplet_classification_im/").exists():
        return
    Path(f"{quadrant_dir}/droplet_classification_im/").mkdir(exist_ok=True)

    tifs = list(sorted(get_files_matching_regex(quadrant_dir, "(?m)^\d+_\d+\.tif$")))
    for root, fname in tqdm(tifs):
        detection_pkl_pth = f"{quadrant_dir}/droplet_detection_pkl/{os.path.splitext(fname)[0]}_droplets.pkl"
        classification_pkl_pth = f"{quadrant_dir}/droplet_classification_pkl/{os.path.splitext(fname)[0]}_classified.pkl"

        with open(detection_pkl_pth, "rb") as fp:
            dets = pkl.loads(fp.read())
        try:
            with open(classification_pkl_pth, "rb") as fp:
                classes = pkl.loads(fp.read())
        except FileNotFoundError:
            continue

        cur_fname = f"{root}/{fname}"
        cur_im = np.array(Image.open(cur_fname))

        classes = {idx: cls for idx, cls in classes}
        retained_dets = []
        retained_cls = []
        for cur in dets:
            if cur[0] in classes:
                retained_dets.append(cur)
                retained_cls.append((cur[0], classes[cur[0]]))

        cur_im = normalize_image(cur_im, (0.1, 99.9)).astype(np.uint8)
        cur_im_circles = draw_circles(
            [((x, y), r) for _, x, y, r in retained_dets],
            cur_im,
            labels=[f"{yy[0:3]} ({xx})" for xx, yy in retained_cls],
        )
        Image.fromarray(cur_im_circles).save(
            f"{quadrant_dir}/droplet_classification_im/{os.path.splitext(fname)[0]}_classification.tif"
        )