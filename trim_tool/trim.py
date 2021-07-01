from pydub import AudioSegment
import os
import csv
import sys




def read_csv(url):  
    '''
    read_csv: open a csv file and returns its contents as an array
    
    input:
        - url: csv file path
    output:
        - content: parsed contend in rows
        example: [['1.857333333', 'Kick'], ['1.952000000', 'Kick'], ['2.813666667', 'Kick'], ['2.878875000', 'Kick'],
    '''
    with open(url, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        content = []
        for row in spamreader:
            row_content = ', '.join(row)
            row_content = row_content.replace(';',',') #participant csv use ; instead of , 
            column_content = [x.strip() for x in row_content.split(',')]
            content.append(column_content)
        return content 

def dataset_log(folder,url,participant_data):
    '''
    dataset_log: creates/update the log of the dataset. The log consists on a csv file that provides metadata for each sound of the dataset.

    input:
        - folder: root folder of the dataset
        - url: dir of the created sound
        - participant_data: data of the contribution
    output:
        - dataset_log: creates or updates dataset log.
    '''
    # Initials
    # Gender
    # Age
    # Microphone
    
    audio = url
    initials = ""
    gender = ""
    age = ""
    microphone = ""
    owner = ""
    
    for data in participant_data:
        if data[0] == 'Initials':
            initials = data[1]
        elif data[0] == 'Gender':
            gender = data[1]
        elif data[0] == 'Age':
            age = data[1]
        elif data[0] == 'Microphone':
            microphone = data[1]
        elif data[0] == 'Owner':
            owner = data[1]

    if owner == "":
        owner = "VM"

    row = [url, owner, initials, age, gender, microphone]

    with open(folder+'/dataset_log.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)

  
def trim_audio(url,folder,csv_file,file_name,participant_data):
    '''
    trim_audio: from the path of an audio (url) and its annotation (csv_file), trims the audio according to this annotation. 
    Organizes the sounds in different folders according to their class (Kick,Snare,HH...)
    
    input:
        - url: wav file path
        - folder: parent folder where data is stored, by default is "Dataset"
        - csv_file: array containing the annotation. (output of read_csv)
        - file_name: file name of the sound
        - participant_data: relevant data about the contribution/sound (initials,gender,age,micro...). 
                                A csv will be created relating each sound to its contributor.
    output:
        - trimmed files: trimmed files organized in folders
    '''
    original_audio = AudioSegment.from_wav(url)    

    #for range iterates over csv file in pairs (annotation csv must be even!)
    for i in range(0,len(csv_file),2):
        index = 1 

        t1 = float(csv_file[i][0]) #get onset
        t2 = float(csv_file[i+1][0]) #get offset
        
        t1 = t1 * 1000 #Works in milliseconds
        t2 = t2 * 1000
        trimmed_audio = original_audio[t1:t2] #get samples in that region
        
        #store trimmed audio locally.
        if not os.path.exists(folder+'/'+csv_file[i][1]):
            os.makedirs(folder+'/'+csv_file[i][1]) #It creates a folder of the class i.e. Kick, if not exists.
        while os.path.exists(folder+'/'+csv_file[i][1]+'/'+csv_file[i][1]+str(index)+'.wav'):
            index += 1 #check if a file with the same name exists, and update the index. HH1.wav , HH2.wav...
        else:
            trimmed_audio_url = folder+'/'+csv_file[i][1]+'/'+csv_file[i][1]+str(index)+'.wav'
            trimmed_audio.export(trimmed_audio_url,format="wav") #Exports to a wav file in the current path.
            dataset_log(folder,trimmed_audio_url,participant_data)
            #print('File generated:' + trimmed_audio_url)
            
def trim_files(startpath):
    '''
    
    trim_files (main): given a directory, goes through all the folders/files in the directory to find wav(raw audios) that match with csv files (annotation) and then call the trim_audio method.
    A wav file matches a csv file if both have the same name!!!

    input:
        - startpath: root folder with all raw audios
    output:
        - trimmed files: trimmed files organized in folders
    '''
    dataset_dir = '../TDataset'

    #create root folder
    if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

    #Create dataset_log.csv file where all the metadata will be located.
    with open(dataset_dir+'/dataset_log.csv', 'w', newline='') as f:
        # create the csv writer
        writer = csv.writer(f)
        # write a row to the csv file
        header = ['Audio', 'Owner', 'Ps. Initials', 'Ps. Age', 'Ps. Gender', 'Ps. Microphone']
        writer.writerow(header)
        # close the file
        f.close()

    for root, dirs, files in os.walk(startpath):
        folder = '/' + os.path.basename(root) + '/'
        csv_output = []
        participant_data = []

        if os.path.exists(startpath+folder+"participant_data.csv"):
            participant_data = read_csv(startpath+folder+"participant_data.csv") #get participant data csv

        # #TODO: Optimize parsing csv and its wav (currently double for...)
        for f in files:
            if f.endswith('.csv'):
                csv_output = read_csv(startpath+folder+f)
                file_name = f.split('.')[0] #csv file name, must match wav file_name. 
                for w in files:
                    if w.endswith('.wav') and w.startswith(file_name):
                        trim_audio(startpath+folder+w,dataset_dir,csv_output,file_name,participant_data)
     
     
trim_files('../RawDataset')