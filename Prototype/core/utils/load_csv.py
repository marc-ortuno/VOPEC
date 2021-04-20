import csv
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

def load_groundtruth(path):
    annotation = read_csv(path)
    groundtruth = []
    for i in range(0,len(annotation),2):
        classtype = annotation[i][1] #get class onset
        groundtruth.append(classtype)
    return groundtruth

def load_annotation(path):
    return read_csv(path)