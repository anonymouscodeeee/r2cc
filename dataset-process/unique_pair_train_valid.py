import json
import math
import os

file_name_path = [
    "dataset/java/type_1.json",
    "dataset/java/type_2.json",
    "dataset/java/type_vst3.json",
    "dataset/java/type_st3.json",
    "dataset/java/type_mt3.json",
    "dataset/java/type_wt3_t4_04_05.json"
]

write_file_path_base_dir = "dataset/java"


# write the training dataset
all_train_code_pair = []
all_train_count = 0
with open(os.path.join(write_file_path_base_dir, "train_unique.json"), 'w', encoding='utf-8') as f_write:
    for i in range(len(file_name_path)):
        train_unique_pairs_dict = {}
        with open(file_name_path[len(file_name_path)-1-i], 'r', encoding='utf-8') as f_read:
            train_unique_pairs_dict = {}

            lines = f_read.readlines()
            for idx, line in enumerate(lines[:math.ceil(len(lines) * 0.8)]):
                pair_dict = json.loads(line)
                all_train_code_pair.append(pair_dict)

                all_train_count += 1
        f_read.close()
        print(f"train pairs: {file_name_path[len(file_name_path)-1-i].split('/')[-1]} - {idx} has load...")

    for j in range(len(all_train_code_pair)):
        json_str = json.dumps(all_train_code_pair[j])
        f_write.write(json_str + "\n")
    print(f"all train: {all_train_count} pairs has written to train_unique.json")
f_write.close()


# write the valid dataset
all_valid_code_pair = []
all_valid_count = 0
with open(os.path.join(write_file_path_base_dir, "valid_unique.json"), 'w', encoding='utf-8') as f_write:
    for i in range(len(file_name_path)):
        valid_unique_pairs_dict = {}
        with open(file_name_path[len(file_name_path)-1-i], 'r', encoding='utf-8') as f_read:
            valid_unique_pairs_dict = {}

            lines = f_read.readlines()
            for idx, line in enumerate(lines[math.ceil(len(lines) * 0.8): math.ceil(len(lines) * 0.9)]):
                pair_dict = json.loads(line)
                all_valid_code_pair.append(pair_dict)

                all_valid_count += 1
        f_read.close()
        print(f"train pairs: {file_name_path[len(file_name_path)-1-i].split('/')[-1]} - {idx} has load...")

    for j in range(len(all_valid_code_pair)):
        json_str = json.dumps(all_valid_code_pair[j])
        f_write.write(json_str + "\n")
    print(f"all valid: {all_valid_count} pairs has written to valid_unique.json")
f_write.close()
