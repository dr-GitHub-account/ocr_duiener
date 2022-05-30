import easyocr
import numpy as np
import os
import json

# 实例化 EasyOCR Reader
reader = easyocr.Reader(['ch_sim','en']) # this needs to run only once to load the model into memory
# 调用接口直接得到结果
result = reader.readtext('/home/user/xiongdengrui/ocr_cluener/ocr/EasyOCR/examples/news_test_3.jpg')
print(result)
print(" ")

# print('np.shape(result)', np.shape(result))
# print('type(result)', type(result))
# # np.shape(result) (29, 3)
# # type(result) <class 'list'>
final_result = []
for i, initial_sequence in enumerate(result):
    continue_flag = 0
    # print(i, initial_sequence)
    same_sequence = [initial_sequence[1]]
    for j in range(i + 1, len(result)):
        following_sequence = result[j]
        # if i == 18:
        #     print(following_sequence[0], np.shape(following_sequence[0]), type(following_sequence[0]))
        #     print(np.array(following_sequence[0])[:, 1], sum(np.array(following_sequence[0])[:, 1]))
        if sum(np.array(following_sequence[0])[:, 1]) - sum(np.array(initial_sequence[0])[:, 1]) < 10:
            same_sequence.append(following_sequence[1])
    if i == 0:
        final_result.append(same_sequence)
    for final_sequence in final_result:
        if set(same_sequence) <= set(final_sequence):
            continue_flag = 1
            break
    if continue_flag == 1:
        continue
    final_result.append(same_sequence)
final_result_str = []
for final_sequence in final_result:
    # print(final_sequence)
    final_result_str.append(''.join(final_sequence))
print(final_result_str, np.shape(final_result_str))

json_name = '/home/user/xiongdengrui/ocr_cluener/pytorch_version/datasets/cluener/test.json'
file = open(file = json_name, mode = "w", encoding = "utf-8")
for final_sequence_str_id, final_sequence_str in enumerate(final_result_str):
    file.write(json.dumps({"id": final_sequence_str_id, "text": final_sequence_str}, ensure_ascii = False) + "\n")
print("ocr result saved at:", json_name)
print("begin ner")


