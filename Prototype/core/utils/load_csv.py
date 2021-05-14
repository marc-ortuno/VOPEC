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
            row_content = row_content.replace(';', ',')  # participant csv use ; instead of ,
            column_content = [x.strip() for x in row_content.split(',')]
            content.append(column_content)
        return content


def load_groundtruth(path):
    annotation = read_csv(path)
    groundtruth = []
    for i in range(0, len(annotation), 2):
        classtype = annotation[i][1]  # get class onset
        groundtruth.append(classtype)
    return groundtruth


def load_annotation(path):
    return read_csv(path)


def get_prediction_time_instants(onset_location, prediction, sr):
    time_list = []
    last_value = 0
    prediction_list = []
    current_index = 0
    for i in range(0, len(onset_location)):
        if onset_location[i] == 1 and last_value == 0:
            time_list.append(i / sr)
        elif onset_location[i] == 0 and last_value == 1:
            time_list.append(i / sr)
        last_value = onset_location[i]

    for j in range(0, len(prediction)):
        prediction_list.append([time_list[j + current_index], prediction[j]])
        prediction_list.append([time_list[j + current_index + 1], prediction[j]])
        current_index += 1

    return prediction_list
