import os
import sys
base_dir = os.getcwd()
sys.path.insert(0, base_dir)
import model
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2, yaml, copy
from easydict import EasyDict as edict
import ctools, gtools
import argparse

# def gazeto3d(gaze):
#     assert gaze.size == 2, "The size of gaze must be 2"
#     gaze_gt = np.zeros([3])
#     gaze_gt[0] = -np.cos(gaze[0]) * np.sin(gaze[1])
#     gaze_gt[1] = -np.sin(gaze[0])
#     gaze_gt[2] = -np.cos(gaze[0]) * np.cos(gaze[1])
#     return gaze_gt

def gazeto3d(gaze):
    x = -torch.cos(gaze[:, 0]) * torch.sin(gaze[:, 1])
    y = -torch.sin(gaze[:, 0])
    z = -torch.cos(gaze[:, 0]) * torch.cos(gaze[:, 1])
    gaze = torch.stack([x, y, z], dim=1)
    return gaze

def main(train, test):

    # =================================> Setup <=========================
    reader = importlib.import_module("reader." + test.reader)
    torch.cuda.set_device(test.device)

    data = test.data
    load = test.load
 

    # ===============================> Read Data <=========================
    if data.isFolder: 
        data, _ = ctools.readfolder(data) 

    dataset = reader.loader(data, 32, num_workers=4, shuffle=False)

    modelpath = os.path.join(train.save.metapath,
                                train.save.folder, f"checkpoint/")
    
    logpath = os.path.join(train.save.metapath,
                                train.save.folder, f"{test.savename}")

  
    if not os.path.exists(logpath):
        os.makedirs(logpath)

    # =============================> Test <=============================

    begin = load.begin_step; end = load.end_step; step = load.steps

    for saveiter in range(begin, end+step, step):

        print(f"Test {saveiter}")

        net = model.Model()

        statedict = torch.load(
                        os.path.join(modelpath, 
                            f"Iter_{saveiter}_{train.save.model_name}.pt"), 
                        map_location={f"cuda:{train.device}": f"cuda:{test.device}"}
                    )

        net.cuda(); net.load_state_dict(statedict); net.eval()

        length = len(dataset); accs0 = 0; accs2 = 0; count = 0

        logname = f"{saveiter}.log"

        outfile0 = open(os.path.join(logpath, f"cam0_" + logname), 'w')
        outfile0.write("name results gts acc\n")
 
        outfile2 = open(os.path.join(logpath, f"cam2_" + logname), 'w')
        outfile2.write("name results gts acc\n")
        
       

        with torch.no_grad():
            for j, (data, label) in enumerate(dataset):

                for key in data:
                    if key != 'name': data[key] = data[key].cuda()

                for key in label:
                    if key != 'name': label[key] = label[key].cuda()

                # names = label["name"]#eth
                names = data["name"]#eve

                cams = data['cams']

                cam1 = cams[:, 0:3, :]
                cam2 = cams[:, 3:6, :]

                gts0 = label['gaze'][:,0,:]
                gts2 = label['gaze'][:,1,:]
           
                gaze0, gaze2  = net(data)
                #convert to 3D
                gaze3d0 = gazeto3d(gaze0)
                gaze3d2 = gazeto3d(gaze2)
                gts0 = gazeto3d(gts0)
                gts2 = gazeto3d(gts2)
                #convert to the world coordinate system
                gaze0 = torch.einsum('ijk,ik->ij', [cam1, gaze3d0])
                gaze2 = torch.einsum('ijk,ik->ij', [cam2, gaze3d2])
                gts0 = torch.einsum('ijk,ik->ij', [cam1, gts0])
                gts2 = torch.einsum('ijk,ik->ij', [cam2, gts2])

                gts = gts0
                gazes = gaze0
                # gazes2 = gaze2

                # for k , (gaze0, gaze2) in enumerate(zip(gazes,gazes2)):
                for k, gaze in enumerate(gazes):
                    gaze0 = gaze.cpu().detach().numpy()
                    gt2 = gts.cpu().numpy()[k]
                    # gaze2 = gaze2.cpu().detach().numpy()
                    # gaze=(gaze0+gaze2)/2 # World coordinate system

                    count += 1                

                    # acc = gtools.angular(gazeto3d(gaze), gazeto3d(gt))
                    acc = gtools.angular(gaze0, gt2)
                    accs0 += acc
            
                    name = [names[k]]
                    gaze = [str(u) for u in gaze0]
                    gt = [str(u) for u in gt2]
                    log = name + [",".join(gaze)] + [",".join(gt)] + [str(acc)]
                    outfile0.write(" ".join(log) + "\n")


                gts = gts2
                gazes = gaze2

                for k, gaze in enumerate(gazes):

                    gaze = gaze.cpu().detach().numpy()
                    gt = gts.cpu().numpy()[k]


                    # acc = gtools.angular(gazeto3d(gaze), gazeto3d(gt))
                    acc = gtools.angular(gaze, gt)
                    accs2 += acc

                    name = [names[k]]
                    gaze = [str(u) for u in gaze]
                    gt = [str(u) for u in gt]
                    log = name + [",".join(gaze)] + [",".join(gt)] + [str(acc)]
                    outfile2.write(" ".join(log) + "\n")


            loger = f"[{saveiter}] Total Num: {count}, cam0: {accs0/count}， cam2: {accs0/count}"
            outfile0.write(loger)
            outfile2.write(loger)
            print(loger)
        outfile0.close()
        outfile2.close()

if __name__ == "__main__":

    # Read model from train config and Test data in test config.
    train_conf = edict(yaml.load(open(sys.argv[1]), Loader=yaml.FullLoader))

    test_conf = edict(yaml.load(open(sys.argv[2]), Loader=yaml.FullLoader))

    print("=======================>(Begin) Config of training<======================")
    print(ctools.DictDumps(train_conf))
    print("=======================>(End) Config of training<======================")
    print("")
    print("=======================>(Begin) Config for test<======================")
    print(ctools.DictDumps(test_conf))
    print("=======================>(End) Config for test<======================")

    main(train_conf.train, test_conf.test)

 
