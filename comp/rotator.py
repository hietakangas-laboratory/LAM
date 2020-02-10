# -*- coding: utf-8 -*-
"""
Rotate coords around origin to properly orientate for LAM vector creation.

USAGE:
-----
    Input degDict, CHANNELS, MAKEPLOTS, CHANGETYPE -settings below, then run
    the script.

Args:
----
    path - pathlib.Path:
        Path to the root directory of LAM-hierarchically structured sample
        folder, i.e. samples at root and channels at subfolders.

    fsavepath - pathlib.Path:
        The directory where the rotated data is saved. If same as 'path', will
        overwrite original data.

    degDict - dict {str: int}:
        Keys indicate names of sample files that will be rotated clock-wise the
        amount of degrees given by value.

    CHANNELS - list [str]:
        List of channels to be rotated.

    MAKEPLOTS - bool:
        Make plots to show if the samples have been correctly rotated.

Created on Thu Sep  5 11:34:24 2019
@author: artoviit
"""
import pandas as pd
import pathlib as pl
import math
import matplotlib.pyplot as plt
import os


path = pl.Path(r"E:\Code_folder\R1_R2")
fsavepath = pl.Path(r"E:\Code_folder\R1_R2\New folder")

# Input samples to be rotated and rspective number of degrees to rotate
# clock-wise
degDict = {"Fed_Fed2": 45, "Fed_Fed4": -45,
           "CtrlYS_S3B": -100, "CtrlYS_S4B": -45,
           "CtrlYS_S3A": -100, "CtrlYS_S1B": 45}
# Channels to rotate
CHANNELS = ["DAPI"]
MAKEPLOTS = True


def ROTATE(path, fsavepath, degDict, CHANNELS, MAKEPLOTS):
    """Rotate all designated channels and samples."""
    def rotate_around_point(x, y, radians, origin=(0, 0)):
        """Rotate a point around a given point."""
        offset_x, offset_y = origin
        adjusted_x = (x - offset_x)
        adjusted_y = (y - offset_y)
        cos_rad = math.cos(radians)
        sin_rad = math.sin(radians)
        qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
        qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y
        return qx, qy

    def change_to_csv(path, filetype=".tif"):
        search = "*{}".format(filetype)
        pathlist = list(path.glob(search))
        for file in pathlist:
            name = str(file.stem).split('.')[0]
            rename = name+".csv"
            rp = pl.Path(file)
            os.rename(rp, rp.parent.joinpath(rename))

    def make_plots(data1, data2, samplename, channel, savepath):
        fig, ax = plt.subplots(2, 1, figsize=(10, 10))
        ax[0].scatter(x=data1.loc[:, "Position X"],
                      y=data1.loc[:, "Position Y"])
        ax[1].scatter(x=data2.loc[:, "Position X"],
                      y=data2.loc[:, "Position Y"])
        ax[0].set_aspect('equal', adjustable='box')
        ax[1].set_aspect('equal', adjustable='box')
        fig.suptitle("{}\n{}".format(samplename, channel))
        filepath = savepath.joinpath("{}_{}.png".format(samplename, channel))
        fig.savefig(filepath, format="png")
        plt.close()

    for samplepath in [p for p in path.iterdir() if 'Analysis Data' not in
                       p.name and p != fsavepath]:
        samplename = str(samplepath.name)
        for channel in CHANNELS:
            if samplename in degDict.keys():
                chpath = next(samplepath.glob('*{}*'.format(channel)))
                print(samplename, channel)
                samplepos = next(chpath.glob("*Position.csv"))
                try:
                    data = pd.read_table(samplepos, index_col=False, header=2,
                                         sep=',')
                    x = data.loc[:, "Position X"]
                except FileNotFoundError:
                    print("File not found")
                    continue
                except (AttributeError, ValueError):
                    print("read_csv Failed")
                    data = pd.read_csv(samplepos, index_col=False)
                data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
                degs = degDict.get(samplename)
                rads = math.radians(degs)
                orgData = data.copy()
                for i, row in data.iterrows():
                    x = row.at["Position X"]
                    y = row.at["Position Y"]
                    x, y = rotate_around_point(x, y, rads)
                    data.at[i, "Position X"] = x
                    data.at[i, "Position Y"] = y
                fsavepath.mkdir(exist_ok=True)
                if MAKEPLOTS:
                    make_plots(orgData, data, samplename, channel, fsavepath)
                strparts = str(samplepos).split("\\")
                newpath = fsavepath.joinpath(strparts[-4], strparts[-3],
                                             strparts[-2], "Position.csv")
                newpath.parent.mkdir(parents=True, exist_ok=True)
                try:
                    newpath.unlink()
                except FileNotFoundError:
                    pass
                with open(newpath, 'a') as f:
                    f.write('\n')
                    f.write('Position\n')
                    f.write('='*data.shape[1]+'\n')
                    data.to_csv(f, index=False, line_terminator=',\n')


if __name__ == '__main__':
    ROTATE(path, fsavepath, degDict, CHANNELS, MAKEPLOTS)
