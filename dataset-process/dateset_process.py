import json
import random

source_code_base_dir = "dataset/java/bigclonebench"
read_file_path = "dataset/java/train_type_mt3_055_06.json"
write_file_path = "dataset/java/train_type_mt3_055_06.json"

random.seed(1234)
with open(write_file_path, 'w', encoding='utf-8') as f_write:
    all_code_pair = []
    count = 0
    with open(read_file_path, 'r', encoding='utf-8') as f_read:
        lines = f_read.readlines()
        for line in lines:
            pair_dict = json.loads(line)
            all_code_pair.append(pair_dict)
            print("loading data: " + str(count))
            count += 1
    f_read.close()

    all_code_pair_index = list(range(len(all_code_pair)))

    code_pair_is_used = [1] * len(all_code_pair)
    code_pair_is_used_dict = dict(zip(all_code_pair_index, code_pair_is_used))

    negative_is_used = [1] * len(all_code_pair)
    negative_is_used_dict = dict(zip(all_code_pair_index, negative_is_used))

    count = 0

    for i in range(len(all_code_pair)):
        is_continue_flag = 0
        while code_pair_is_used_dict[i] == 1:
            if is_continue_flag >= 1000:
                break
            func_1_id = all_code_pair[i]['func_1_id']
            func_2_id = all_code_pair[i]['func_2_id']
            func_true_functionality_id = all_code_pair[i]['functionality_id']

            random_choose_pair_index = random.randint(0, len(all_code_pair) - 1)
            print(count)
            count += 1
            random_choose_pair_1 = all_code_pair[random_choose_pair_index]['func_1_id']
            random_choose_pair_2 = all_code_pair[random_choose_pair_index]['func_2_id']
            func_random_functionality_id = all_code_pair[random_choose_pair_index]['functionality_id']
            if negative_is_used_dict[random_choose_pair_index] == 1 and \
                    func_true_functionality_id != func_random_functionality_id and \
                    func_1_id != random_choose_pair_1 and func_1_id != random_choose_pair_2 and \
                    func_2_id != random_choose_pair_1 and func_2_id != random_choose_pair_2:
                all_code_pair[i]['func_3_id_negative'] = all_code_pair[random_choose_pair_index]['func_2_id']
                all_code_pair[i]['func_3_source_code_negative'] = all_code_pair[random_choose_pair_index]['func_2_source_code']
                all_code_pair[i]['func_3_functionality_id'] = func_random_functionality_id
                negative_is_used_dict[random_choose_pair_index] = 0
                code_pair_is_used_dict[i] = 0
            is_continue_flag += 1
    print("negative pair construct done!")

    for i in range(len(all_code_pair)):
        if len(all_code_pair[i]) == 18:
            json_str = json.dumps(all_code_pair[i])
            f_write.write(json_str + "\n")
    print("negative pair has written to output file")
f_write.close()
