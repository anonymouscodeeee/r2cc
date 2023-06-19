import json
import math

unique_file_name_path = [
    "dataset/java/type_1.json",
    "dataset/java/type_2.json",
    "dataset/java/type_vst3.json",
    "dataset/java/type_st3.json",
    "dataset/java/type_mt3.json",
    "dataset/java/type_wt3_t4_04_05.json"
]

per_type_test_file_name_path = [
    "dataset/java/type_1.json",
    "dataset/java/type_2.json",
    "dataset/java/type_vst3.json",
    "dataset/java/type_st3.json",
    "dataset/java/type_mt3.json",
    "dataset/java/type_wt3_t4_04_05.json"
]


# write the training dataset
for i in range(len(unique_file_name_path)):
    all_train_code_pair = []
    with open(per_type_test_file_name_path[len(per_type_test_file_name_path)-1-i], 'w', encoding='utf-8') as f_write:
        train_unique_pairs_dict = {}
        with open(unique_file_name_path[len(unique_file_name_path)-1-i], 'r', encoding='utf-8') as f_read:
            train_unique_pairs_dict = {}

            lines = f_read.readlines()
            for idx, line in enumerate(lines[math.ceil(len(lines) * 0.9):]):
                pair_dict = json.loads(line)
                all_train_code_pair.append(pair_dict)
                if idx >= 999:
                    break
        f_read.close()

        for j in range(len(all_train_code_pair)):
            json_str = json.dumps(all_train_code_pair[j])
            f_write.write(json_str + "\n")
        print(f"test pairs: {unique_file_name_path[len(per_type_test_file_name_path)-1-i].split('/')[-1]} - {idx} has written {per_type_test_file_name_path[len(per_type_test_file_name_path)-1-i]}")
    f_write.close()
