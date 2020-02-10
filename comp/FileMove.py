# -*- coding: utf-8 -*-
"""
Move and rename image files after splitting channels and focal planes.

USAGE:
-----
    Used subsequently to ImageJ "Macro_Imaris File Converter.ijm". Takes path
    to directory containing the images, renames them to have sample group pre-
    fixes, and moves them to sample-specific folders.

    To use, remove the number signs (#) from before the required functions at
    the end of the file.

Vars:
----
    Path - pathlib.Path:
        Path to the directory that contains the images.

    nameDict - dict {str: pathlib.Path}:
        Determine the prefixes that are given to the images when renamed.
            Key = prefix
            Value = path to target images

    ext - str:
        Determine the file extension that is searched for.

Funcs:
-----
    createFolderAndMove:
        Takes in Path-variable and moves each sample's files found at the
        directory into sample-specific directories.

    Rename:
        Finds images at the paths defined by nameDict's values and gives them a
        prefix '<key>_', e.g. xyz.tif -> Starved_xyz.tif.

    rmvRename:
        Removes prefixes made by Rename-function.

Created on Fri Sep  6 16:47:08 2019
@author: artoviit
"""

import pathlib as pl
import shutil
import os

Path = pl.Path(r"D:\230819_Notch\Starved_split")
nameDict = {"Fed": pl.Path(r"D:\230819_Notch\Fed_split"),
            "Starved": pl.Path(r"D:\230819_Notch\Starved_split")
            }
ext = 'tif'


def createFolderAndMove(Path):
    """Move image files at path to new sample-specific directories."""
    filepaths = Path.glob("*.{}".format(ext))
    for path in filepaths:
        print(path)
        strlist = str(path.stem).split("_")
        name = strlist[0]
        chan = strlist[2]
        folder = "{}_{}".format(name, chan)
        folderpath = Path.joinpath(folder)
        folderpath.mkdir(exist_ok=True)
        filename = path.name
        shutil.move(path, folderpath.joinpath(filename))


def Rename(nameDict):
    """Rename all images and samplefolders found within the path."""
    for key in nameDict.keys():
        filepath = nameDict.get(key)
        prex = str(key+"_")
        for d in filepath.iterdir():
            files = d.glob("*.{}".format(ext))
            for file in files:
                if prex not in str(file.name):
                    name = file.name
                    fullname = str(prex+name)
                    os.rename(file, d.joinpath(fullname))
            dirname = d.name
            if prex not in str(dirname):
                dirNew = str(prex+dirname)
                os.rename(d, filepath.joinpath(dirNew))


def rmvRename(nameDict):
    """Remove any renames made by the Rename()-function."""
    for key in nameDict.keys():
        filepath = nameDict.get(key)
        prex = str(key+"_")
        for d in filepath.iterdir():
            dirname = str(d.name)
            dirNew = dirname.replace(prex, '')
            os.rename(d, filepath.joinpath(dirNew))

# Rename(nameDict)
# rmvRename(nameDict)
# createFolderAndMove(Path)
