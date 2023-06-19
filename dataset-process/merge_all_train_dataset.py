import json
import math
import os

file_name_path = [
    "dataset/java/type_1_train.json",
    "dataset/java/type_2_train.json",
    "dataset/java/type_vst3_train.json",
    "dataset/java/type_st3_train.json",
    "dataset/java/train_type_mt3_06_07_train.json",
    "dataset/java/train_type_mt3_055_06_train.json"
]

write_file_path_base_dir = "dataset/java/"

# write the training dataset
all_train_code_pair = []
all_train_count = 0

for i in range(len(file_name_path)):
    with open(file_name_path[i], 'r', encoding='utf-8') as f_read:
        lines = f_read.readlines()
        for idx, line in enumerate(lines):
            pair_dict = json.loads(line)
            all_train_code_pair.append(pair_dict)
            all_train_count += 1
    f_read.close()
    print(f"train pairs: {file_name_path[i]} : {idx} has written to train.json")

with open(os.path.join(write_file_path_base_dir, "train_all_055not045.json"), 'w', encoding='utf-8') as f_write:
    for j in range(len(all_train_code_pair)):
        json_str = json.dumps(all_train_code_pair[j])
        f_write.write(json_str + "\n")
    print(f"all train: {all_train_count} pairs has written to train.json")
f_write.close()


# write the valid dataset
all_valid_code_pair = []
all_valid_count = 0
with open(os.path.join(write_file_path_base_dir, "valid_055not045.json"), 'w', encoding='utf-8') as f_write:
    for i in range(len(file_name_path)):
        count = 0
        valid_unique_pairs_dict = {}
        with open(file_name_path[len(file_name_path)-1-i], 'r', encoding='utf-8') as f_read:
            valid_unique_pairs_dict = {}

            lines = f_read.readlines()
            for line in lines[math.ceil(len(lines) * 0.8): math.ceil(len(lines) * 0.9)]:
                pair_dict = json.loads(line)
                all_valid_code_pair.append(pair_dict)

                if pair_dict['func_1_id'] not in list(valid_unique_pairs_dict.keys()):
                    valid_unique_pairs_dict[pair_dict['func_1_id']] = pair_dict['func_2_id']
                count += 1
                all_valid_count += 1
        f_read.close()
        print(f"train pairs: {file_name_path[len(file_name_path)-1-i].split('/')[-1]} - {count} has written to train.json")

    for j in range(len(all_valid_code_pair)):
        json_str = json.dumps(all_valid_code_pair[j])
        f_write.write(json_str + "\n")
    print(f"all valid: {all_valid_count} pairs has written to train.json")
f_write.close()
