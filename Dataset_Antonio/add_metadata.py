import os
import csv
import sys

def create_participant_data(path,initials,mic):
       
    with open(path+'/participant_data.csv', 'w', newline='') as f:
        # create the csv writer
        writer = csv.writer(f)

        # write a row to the csv file
        Initials = ['Initials',initials]
        Gender = ['Gender', '*']
        Age = ['Age', '*']
        Microphone = ['Microphone', mic]
        Owner = ['Owner','AFR']

        writer.writerow(Initials)
        writer.writerow(Gender)
        writer.writerow(Age)
        writer.writerow(Microphone)
        writer.writerow(Owner)
        
        # close the file
        f.close()

def create_participant_antonio(startpath):

    for root, dirs, files in os.walk(startpath):

        if len(root.split('/')) > 2:
            print(root)
            mic_folder = root.split('/')[1]
            mic = "*"
            if "2" in mic_folder:
                mic = "AKG c4000b"
            else:
                mic = "iPad"

            initials = root.split('/')[2]
            create_participant_data(root,initials,mic)

     
     
create_participant_antonio('./')    