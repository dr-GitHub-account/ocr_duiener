import json
import numpy as np
from tqdm import tqdm

original_file_path = "/home/user/xiongdengrui/ocr_duiener/pytorch_version/datasets/duiener/train_ext.json"
destination_file_path = "/home/user/xiongdengrui/ocr_duiener/pytorch_version/datasets/duiener/ner_train_ext.json"

destination_file = open(file = destination_file_path, mode = "w", encoding = "utf-8")

# open file, get load_ori(_io.TextIOWrapper type)
with open(original_file_path,'r') as load_ori:
    # read the lines
    for line in load_ori.readlines():
        line_dict = eval(line)
        text = line_dict["text"]
        label = line_dict["label"]
        Work = label["Work"]
        Movie = label["Movie"]
        print(list(Work.keys())[0])
        print(list(Movie.keys())[0])
        Work[list(Work.keys())[0]] = [[text.find(list(Work.keys())[0]), text.find(list(Work.keys())[0]) + len(list(Work.keys())[0]) - 1]]
        Movie[list(Movie.keys())[0]] = [[text.find(list(Movie.keys())[0]), text.find(list(Movie.keys())[0]) + len(list(Movie.keys())[0]) - 1]]
        destination_file.write(json.dumps(line_dict, ensure_ascii = False) + "\n")