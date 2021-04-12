from pydub import AudioSegment
import os
import csv
import sys


def trim_files(startpath,old,new):
    print(startpath)
    for root, dirs, files in os.walk(startpath):
        folder = '/'+ os.path.basename(root) + '/'
        current_path = startpath + folder
        for f in files:
            if f.split('_')[0] == old:
                file_info = f.split('_')[1]
                os.rename(current_path + f,current_path + new + "_" +file_info)
                print(file_info)

     
     
trim_files('../Dataset',"HiHat","HH")    