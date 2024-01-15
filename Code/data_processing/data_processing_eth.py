import os
import h5py
import numpy as np
import cv2

path = "/data/wyl_data/xgaze_224/train"
imo_path = "/data/wyl_data/ETH-Gaze/Image/train"
annoo_path = "/data/wyl_data/ETH-Gaze/Label/train.label"
test = True
file_order=["subject0059.h5","subject0088.h5","subject0010.h5","subject0033.h5","subject0035.h5","subject0038.h5","subject0111.h5","subject0100.h5","subject0007.h5","subject0004.h5","subject0031.h5","subject0067.h5","subject0063.h5","subject0029.h5","subject0101.h5","subject0014.h5","subject0098.h5","subject0095.h5","subject0041.h5","subject0019.h5","subject0036.h5","subject0109.h5","subject0108.h5","subject0099.h5","subject0104.h5","subject0069.h5","subject0039.h5","subject0080.h5","subject0058.h5","subject0092.h5","subject0021.h5","subject0102.h5","subject0106.h5","subject0061.h5","subject0052.h5","subject0081.h5","subject0056.h5","subject0045.h5","subject0040.h5","subject0079.h5","subject0078.h5","subject0044.h5","subject0009.h5","subject0008.h5","subject0048.h5","subject0018.h5","subject0015.h5","subject0065.h5","subject0032.h5","subject0083.h5","subject0026.h5","subject0060.h5","subject0028.h5","subject0051.h5","subject0024.h5","subject0085.h5","subject0050.h5","subject0016.h5","subject0003.h5","subject0006.h5","subject0084.h5","subject0076.h5","subject0055.h5","subject0057.h5","subject0030.h5","subject0073.h5","subject0027.h5","subject0062.h5","subject0090.h5","subject0043.h5","subject0075.h5","subject0107.h5","subject0072.h5","subject0103.h5","subject0013.h5"]
filenames = os.listdir(path)
# filenames.sort()
def custom_sort(filenames):
    return file_order.index(filenames) if filenames in file_order else float("inf")
filenames.sort(key=custom_sort)

if not os.path.exists(imo_path):
    os.makedirs(imo_path)

if not os.path.exists(os.path.dirname(annoo_path)):
    os.makedirs(os.path.dirname(annoo_path))

if not test:
    with open(annoo_path, 'w') as outfile:
        outfile.write("face gaze head origin cam_index frame_index normmat\n")
else:
    with open(annoo_path, 'w') as outfile:
        outfile.write("face head origin cam_index frame_index normmat\n")


def process_person(h5files_path, imo_metapath, sub_id, annoo_metapath, begin_num, test):
    datas = h5py.File(h5files_path, 'r')
    if not test:
        keys = ["cam_index", "face_gaze", "face_head_pose",
            "face_mat_norm", "face_patch", "frame_index"]
    else:
        keys = ["cam_index", "face_head_pose",
            "face_mat_norm", "face_patch", "frame_index"]
    length = datas[keys[0]].shape[0]
    print(f"==> Length: {length}")

    imo_path = os.path.join(imo_metapath, sub_id)
    if not os.path.exists(imo_path):
        os.makedirs(imo_path)

    with open(annoo_metapath, 'a') as outfile:
        for i in range(length):
            img = datas["face_patch"][i,:]
            cv2.imwrite(os.path.join(imo_path, f"{begin_num}.jpg"), img)

            im_path = os.path.join(sub_id, f"{begin_num}.jpg")
            head = ",".join(list(datas["face_head_pose"][i, :].astype("str")))
            norm_mat = ",".join(list(datas["face_mat_norm"][i, :].astype("str").flatten()))
            cam_index = ",".join(list(datas["cam_index"][i, :].astype("str")))
            frame_index = ",".join(list(datas["frame_index"][i, :].astype("str")))
            if not test: 
                gaze = ",".join(list(datas["face_gaze"][i, :].astype("str")))
                outfile.write(f"{im_path} {gaze} {head} {os.path.join(sub_id, str(i)+'.jpg')} {cam_index} {frame_index} {norm_mat}\n")
            else: 
                outfile.write(f"{im_path} {head} {os.path.join(sub_id, str(i)+'.jpg')} {cam_index} {frame_index} {norm_mat}\n")
            begin_num += 1
    datas.close()
    return begin_num

num = 1

for count, filename in enumerate(filenames):
    # print(filenames)
    print(f"Processing.. {filename}, [{count}/{len(filenames)}]")
    sub_id = filename.split(".")[0]
    file_path = os.path.join(path, filename)
    num = process_person(file_path, 
        imo_path,
        sub_id,
        annoo_path,
        num,
        test)
