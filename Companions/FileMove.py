# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 16:47:08 2019

Script for renaming image files after using ImageJ "Macro_Imaris File Converter.ijm"
to split channels and focal planes.

@author: artoviit
"""

import pathlib as pl
import shutil
import os

Path = pl.Path(r"D:\Arto Viitanen\Microscopy\230819_Notch guts for analysis\Starved_split")
nameDict = {"Fed": pl.Path(r"D:\Arto Viitanen\Microscopy\230819_Notch guts for analysis\Fed_split"),
            "Starved": pl.Path(r"D:\Arto Viitanen\Microscopy\230819_Notch guts for analysis\Starved_split")
            }

def createFolderAndMove(Path):
    """ Finds all split image files in the directory, creates a folder for each 
    sample, and moves related files into the folders."""
    filepaths = Path.glob("*.tif")
    for path in filepaths:
        print(path)
        strlist = str(path.stem).split("_")
        name = strlist[0]
        chan = strlist[2]
        folder = "{}_{}".format(name,chan)
        folderpath = Path.joinpath(folder)
        folderpath.mkdir(exist_ok=True)
        filename = path.name
        shutil.move(path, folderpath.joinpath(filename))

def Rename(nameDict):
    """ Renames all images and samplefolders found within the paths indicated 
    by nameDict. The nameDict keys indicate the prefix that is added before the file and folder names. """
    for key in nameDict.keys():
        filepath = nameDict.get(key)
        prex = str(key+"_")
        for d in filepath.iterdir():
            files = d.glob("*.tif")
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
    """ Removes any renames made by the Rename()-function"""
    for key in nameDict.keys():
        filepath = nameDict.get(key)
        prex = str(key+"_")
        for d in filepath.iterdir():
            dirname = str(d.name)
            dirNew = dirname.replace(prex, '')
            os.rename(d, filepath.joinpath(dirNew))

#Rename(nameDict)
#rmvRename(nameDict)
#createFolderAndMove(Path)