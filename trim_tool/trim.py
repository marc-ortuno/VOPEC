from pydub import AudioSegment
import os
import csv
import sys

def read_csv(url):
   with open(url, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    content = []
    for row in spamreader:
        row_content = ', '.join(row) 
        column_content = [x.strip() for x in row_content.split(',')]
        content.append(column_content)
    return content 

  
def trim_audio(url,folder,csv_file,f):
    original_audio = AudioSegment.from_wav(url)    
    file_name = f.split('.')[0] 
    for i in range(0,len(csv_file)):
        index = 1 
        if (i+1) >= len(csv_file):
            t1 = float(csv_file[i][0])
            t2 = original_audio.duration_seconds
        else:
            t1 = float(csv_file[i][0])
            t2 = float(csv_file[i+1][0]) 
        
        t1 = t1 * 1000 #Works in milliseconds
        t2 = t2 * 1000
        trimed_audio = original_audio[t1:t2]
        if not os.path.exists('./'+folder+'/'+csv_file[i][1]):
            os.makedirs('./'+folder+'/'+csv_file[i][1])
        while os.path.exists('./'+folder+'/'+csv_file[i][1]+'/'+csv_file[i][1]+str(index)+'_'+file_name+'.wav'):
            index += 1
        else:
            trimed_audio.export('./'+folder+'/'+csv_file[i][1]+'/'+csv_file[i][1]+str(index)+'_'+file_name+'.wav',format="wav") #Exports to a wav file in the current path.
            print('File generated:' + './'+folder+'/'+csv_file[i][1]+'/'+csv_file[i][1]+str(index)+'_'+file_name+'.wav')
            
def trim_files(startpath):
    print(startpath)
    for root, dirs, files in os.walk(startpath):
        folder = '/' + os.path.basename(root) + '/'
        csv_output = []
        for f in files:
            if f.endswith('.csv'):
                print(folder)
                csv_output = read_csv(startpath+folder+f)
            if f.endswith('.wav'):
                trim_audio(startpath+folder+f,'Dataset',csv_output,f)
     
     
trim_files('../RawDataset/WAV+Annotation')    