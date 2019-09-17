# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 11:34:24 2019

@author: artoviit
"""
import pandas as pd
import pathlib as pl
import math
import matplotlib.pyplot as plt
import os

path = pl.Path(r"P:\h919\hietakangas\Arto\Statistics_DANA\Posterior")
savepath = pl.Path(r"P:\h919\hietakangas\Arto\Statistics_DANA\rotated")

degDict = { "CtrlYS_S1A": 90, "CtrlYS_S1B": 90, "CtrlYS_S2A": 270, "CtrlYS_S2B": 270,
           "CtrlY_S1A": 90, "CtrlY_S2A": 90, "CtrlY_S3A": 90, "CtrlY_S4A": 90,
           "CtrlY_S5A": 90, "CtrlY_S6A": 270, "CtrlY_S8A": 90, "CtrlY_S9A": 90,
           "CtrlY_S9B": 90, "CtrlY_S7A": 270 }

MAKEPLOTS = False
CHANGETYPE = False

def rotate_around_point_highperf(x, y, radians, origin=(0, 0)):
    """Rotate a point around a given point."""
    offset_x, offset_y = origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = math.cos(radians)
    sin_rad = math.sin(radians)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y
    return qx, qy

def change_to_csv(path, filetype = ".tif"):
    search = "*{}".format(filetype)
    pathlist = list(path.glob(search))
    for file in pathlist:
        name = str(file.stem).split('.')[0]
        rename = name+".csv"
        rp = pl.Path(file)
        os.rename(rp, rp.parent.joinpath(rename))
        
def make_plots(data1, data2, samplename, channel, savepath):
    fig, ax = plt.subplots(2, 1, figsize=(3,6))
    ax[0].scatter(x = data1.loc[:, "Position X"], y = data1.loc[:, "Position Y"])
    ax[1].scatter(x = data2.loc[:, "Position X"], y = data2.loc[:, "Position Y"])
    fig.suptitle("{}\n{}".format(samplename, channel))    
    fig.savefig(savepath.joinpath("{}_{}.png".format(samplename,channel)), format="png")

for samplepath in path.iterdir():
    samplename = str(samplepath.stem).split('-')[0]    
    for channel in ["DAPI", "PROS"]:
        path = pl.Path(next(samplepath.glob('*{}*'.format(channel))))
        if CHANGETYPE:
            change_to_csv(path)
        if samplename in degDict.keys():
            samplepos = path.joinpath("Position.csv")
            print(samplepos)
            try:
                data = pd.read_table(samplepos, index_col = False, header=2, 
                                     sep=',')
                x = data.loc[:, "Position X"]
            except FileNotFoundError:
                print("File not found")
                continue
            except:
                print("read_csv Failed")
                data = pd.read_csv(samplepos, index_col = False)
            data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
            degs = degDict.get(samplename)
            rads = math.radians(degs)
            xy = data.loc[:, ["Position X", "Position Y"]]
            xmin, xmax = xy.loc[:,"Position X"].min(), xy.loc[:,"Position X"].max()
            ymin, ymax = xy.loc[:,"Position Y"].min(), xy.loc[:,"Position Y"].max()
            xmed = (xmax-xmin)/2
            ymed = (ymax-ymin)/2
            point = (xmed, ymed)
            orgData = data
            for i, row in data.iterrows():
                x = row.at["Position X"]
                y = row.at["Position Y"]
                x, y = rotate_around_point_highperf(x, y, rads, point)
                data.at[i, "Position X"] = x
                data.at[i, "Position Y"] = y
            if MAKEPLOTS:
                make_plots(orgData, data, samplename, channel, savepath)
            strparts = str(samplepos).split("\\")
            newpath = savepath.joinpath(strparts[-4], strparts[-3], 
                                        strparts[-2], "Position.csv")
            data.to_csv(str(newpath), index = False)