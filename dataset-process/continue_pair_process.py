import json
import os

file_name_path = [
     "dataset/java/train_type_wt3_t4_045_05.json"
]

write_file_path_base_dir = "dataset/java"

all_train_code_pair = []
for i in range(len(file_name_path)):
    with open(file_name_path[i], 'r', encoding='utf-8') as f_read:
        lines = f_read.readlines()
        for idx, line in enumerate(lines):
            paird_dict = json.loads(line)
            all_train_code_pair.append(paird_dict)
            print(idx)
    f_read.close()
    print(f"train pairs: {file_name_path[i].split('/')[-1]} - {idx} has load...")

unique_count = 0
unique_pairs_dict = {}
for i in range(len(file_name_path)):
    with open(file_name_path[i], 'r', encoding='utf-8') as f_read:
        lines = f_read.readlines()
        for idx, line in enumerate(lines):
            pair_dict = json.loads(line)

            if unique_count >= 9646:
                break
            if pair_dict['func_1_id'] not in list(unique_pairs_dict.keys()):
                unique_pairs_dict[pair_dict['func_1_id']] = pair_dict['func_2_id']
                unique_count += 1
            print(idx)
    f_read.close()
    print(f"train pairs: {file_name_path[i].split('/')[-1]} - {idx} has load...")


with open(os.path.join(write_file_path_base_dir, "train_type_wt3_t4_045_05_test_continue.json"), 'w', encoding='utf-8') as f_write:
    for i in range(len(all_train_code_pair)):
        if i > idx and i <= idx + 1000:
            json_str = json.dumps(all_train_code_pair[i])
            f_write.write(json_str + "\n")
f_write.close()
