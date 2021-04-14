from pydub import AudioSegment
import os
import csv
import sys

'''
This script is used to rename the class of some file with structure <class>_<initials>.<extension>
variables:
 - startpath: path where files are located
 - old: old class name of the file
 - mew: new name for the file 

 example:
    input: rename_files('../Dataset',"HiHat","HH")   
    output: HiHat_MOB.wav ---renamed---> HH_MOB.wav
 '''
def rename_files(startpath,old,new):
    print(startpath)
    for root, dirs, files in os.walk(startpath):
        folder = '/'+ os.path.basename(root) + '/'
        current_path = startpath + folder
        for f in files:
            if f.split('_')[0] == old:
                file_info = f.split('_')[1]
                os.rename(current_path + f,current_path + new + "_" +file_info)
                print(file_info)

     
     
rename_files('../Dataset',"HiHat","HH")    