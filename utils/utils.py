import csv
import sys

#log the terminal message in the txt
class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def csv_writer(process_dict, header=None):
    key_list = process_dict.keys()
    value_list = process_dict.values()
    rows = zip(key_list, value_list)
    with open('./test.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            # print(row)
            writer.writerow(row)
    print("数据已经写入成功！！！")

def prediction(output):
    _, pred = output.topk(1, 1, True, True)
    
    return pred[0][0].cpu().numpy()