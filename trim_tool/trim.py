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

  
def trim_audio(url,folder,csv_file,f,file_name):
    original_audio = AudioSegment.from_wav(url)    
    print(file_name)
    for i in range(0,len(csv_file),2):
        index = 1 

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
        #TODO: Optimize parsing csv and its wav (currently double for...)
        for f in files:
            if f.endswith('.csv'):
                csv_output = read_csv(startpath+folder+f)
                file_name = f.split('.')[0]
                for w in files:
                    if w.endswith('.wav') and w.startswith(file_name):
                        trim_audio(startpath+folder+w,'Dataset',csv_output,w,file_name)
     
     
trim_files('../Dataset')    