import argparse
import logging
import os
import pickle
import random
import torch
import json
import numpy as np
import torch.nn.functional as F
import multiprocessing
import math
import faiss
import joblib

from tqdm import tqdm
from model import Model, Second_Stage_Model
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)

from parser import DFG_python, DFG_java, DFG_ruby, DFG_go, DFG_php, DFG_javascript
from parser import (remove_comments_and_docstrings,
                    tree_to_token_index,
                    index_to_code_token,
                    tree_to_variable_index)
from tree_sitter import Language, Parser
from datetime import datetime

logger = logging.getLogger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
cpu_cont = 16

dfg_function = {
    'python': DFG_python,
    'java': DFG_java,
    'ruby': DFG_ruby,
    'go': DFG_go,
    'php': DFG_php,
    'javascript': DFG_javascript
}

# load parsers
parsers = {}
for lang in dfg_function:
    LANGUAGE = Language('parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser, dfg_function[lang]]
    parsers[lang] = parser


# token code and extract dataflow
def extract_dataflow(code, parser, lang):
    tree = parser[0].parse(bytes(code, 'utf8'))
    root_node = tree.root_node
    tokens_index = tree_to_token_index(root_node)
    code = code.split('\n')
    code_tokens = [index_to_code_token(x, code) for x in tokens_index]
    index_to_code = {}
    for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
        index_to_code[index] = (idx, code)
    try:
        DFG, _ = parser[1](root_node, index_to_code, {})
    except:
        DFG = []
    DFG = sorted(DFG, key=lambda x: x[1])
    indexs = set()
    for d in DFG:
        if len(d[-1]) != 0:
            indexs.add(d[1])
        for x in d[-1]:
            indexs.add(x)
    new_DFG = []
    for d in DFG:
        if d[1] in indexs:
            new_DFG.append(d)
    dfg = new_DFG
    return code_tokens, dfg


def extract_code_feature(item, source_code_type):
    js, tokenizer, args = item
    # code
    parser = parsers[args.lang]
    # extract data flow
    code_tokens, dfg = extract_dataflow(js[source_code_type], parser, args.lang)
    code_tokens = [tokenizer.tokenize('@ ' + x)[1:] if idx != 0 else tokenizer.tokenize(x) for idx, x in enumerate(code_tokens)]
    ori2cur_pos = {}
    ori2cur_pos[-1] = (0, 0)
    for i in range(len(code_tokens)):
        ori2cur_pos[i] = (ori2cur_pos[i - 1][1], ori2cur_pos[i - 1][1] + len(code_tokens[i]))
    code_tokens = [y for x in code_tokens for y in x]
    # truncating
    code_tokens = code_tokens[:args.code_length + args.data_flow_length-2-min(len(dfg), args.data_flow_length)]
    code_tokens = [tokenizer.cls_token] + code_tokens[:510] + [tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    position_idx = [i + tokenizer.pad_token_id + 1 for i in range(len(code_tokens))]
    dfg = dfg[:args.code_length + args.data_flow_length - len(code_tokens)]
    code_tokens += [x[0] for x in dfg]
    position_idx += [0 for x in dfg]
    code_ids += [tokenizer.unk_token_id for x in dfg]
    padding_length = args.code_length + args.data_flow_length - len(code_ids)
    position_idx += [tokenizer.pad_token_id] * padding_length
    code_ids += [tokenizer.pad_token_id] * padding_length
    # reindex
    reverse_index = {}
    for idx, x in enumerate(dfg):
        reverse_index[x[1]] = idx
    for idx, x in enumerate(dfg):
        dfg[idx] = x[:-1] + ([reverse_index[i] for i in x[-1] if i in reverse_index],)
    dfg_to_dfg = [x[-1] for x in dfg]
    dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]
    length = len([tokenizer.cls_token])
    dfg_to_code = [(x[0] + length, x[1] + length) for x in dfg_to_code]
    return code_tokens, code_ids, position_idx, dfg_to_code, dfg_to_dfg


def convert_examples_to_features(item):
    # anchor code
    anchor_code_tokens, anchor_code_ids, anchor_position_idx, anchor_dfg_to_code, anchor_dfg_to_dfg = extract_code_feature(item, 'func_1_source_code')

    # positive code clone pair
    positive_code_tokens, positive_code_ids, positive_position_idx, positive_dfg_to_code, positive_dfg_to_dfg = extract_code_feature(item, 'func_2_source_code')

    # negative code clone pair
    negative_code_tokens, negative_code_ids, negative_position_idx, negative_dfg_to_code, negative_dfg_to_dfg = extract_code_feature(item, 'func_3_source_code_negative')

    return InputFeatures(anchor_code_tokens, anchor_code_ids, anchor_position_idx, anchor_dfg_to_code, anchor_dfg_to_dfg,
                         positive_code_tokens, positive_code_ids, positive_position_idx, positive_dfg_to_code, positive_dfg_to_dfg,
                         negative_code_tokens, negative_code_ids, negative_position_idx, negative_dfg_to_code, negative_dfg_to_dfg)


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 anchor_code_tokens, anchor_code_ids, anchor_position_idx, anchor_dfg_to_code, anchor_dfg_to_dfg,
                 positive_code_tokens, positive_code_ids, positive_position_idx, positive_dfg_to_code, positive_dfg_to_dfg,
                 negative_code_tokens, negative_code_ids, negative_position_idx, negative_dfg_to_code, negative_dfg_to_dfg
    ):
        # anchor code pair
        self.anchor_code_tokens = anchor_code_tokens
        self.anchor_code_ids = anchor_code_ids
        self.anchor_position_idx = anchor_position_idx
        self.anchor_dfg_to_code = anchor_dfg_to_code
        self.anchor_dfg_to_dfg = anchor_dfg_to_dfg

        # positive code clone pair
        self.positive_code_tokens = positive_code_tokens
        self.positive_code_ids = positive_code_ids
        self.positive_position_idx = positive_position_idx
        self.positive_dfg_to_code = positive_dfg_to_code
        self.positive_dfg_to_dfg = positive_dfg_to_dfg

        # negative code clone pair
        self.negative_code_tokens = negative_code_tokens
        self.negative_code_ids = negative_code_ids
        self.negative_position_idx = negative_position_idx
        self.negative_dfg_to_code = negative_dfg_to_code
        self.negative_dfg_to_dfg = negative_dfg_to_dfg


class CodeDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None, pool=None):
        self.args = args
        prefix = file_path.split('/')[-1][:-5]
        cache_file = args.output_dir + '/' + prefix + '.pkl'
        if os.path.exists(cache_file):
            self.examples = pickle.load(open(cache_file, 'rb'))
            # self.examples = joblib.load(open(cache_file, 'rb'))
        else:
            self.examples = []
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for idx, line in enumerate(lines):
                    line = line.strip()
                    js = json.loads(line)
                    js['func_1_source_code'] = ' '.join(js['func_1_source_code'].split())
                    js['func_2_source_code'] = ' '.join(js['func_2_source_code'].split())
                    js['func_3_source_code_negative'] = ' '.join(js['func_3_source_code_negative'].split())
                    data.append((js, tokenizer, args))
            self.examples = pool.map(convert_examples_to_features, tqdm(data, total=len(data)))
            pickle.dump(self.examples, open(cache_file, 'wb'))

        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                # anchor code pair
                logger.info("anchor code pair")
                logger.info("anchor_code_tokens: {}".format([x.replace('\u0120', '_') for x in example.anchor_code_tokens]))
                logger.info("anchor_code_ids: {}".format(' '.join(map(str, example.anchor_code_ids))))
                logger.info("anchor_position_idx: {}".format(example.anchor_position_idx))
                logger.info("anchor_dfg_to_code: {}".format(' '.join(map(str, example.anchor_dfg_to_code))))
                logger.info("anchor_dfg_to_dfg: {}".format(' '.join(map(str, example.anchor_dfg_to_dfg))))
                # positive code clone pair
                logger.info("positive code clone pair")
                logger.info("positive_code_tokens: {}".format([x.replace('\u0120', '_') for x in example.positive_code_tokens]))
                logger.info("positive_code_ids: {}".format(' '.join(map(str, example.positive_code_ids))))
                logger.info("positive_position_idx: {}".format(example.positive_position_idx))
                logger.info("positive_dfg_to_code: {}".format(' '.join(map(str, example.positive_dfg_to_code))))
                logger.info("positive_dfg_to_dfg: {}".format(' '.join(map(str, example.positive_dfg_to_dfg))))
                # negative code clone pair
                logger.info("negative code clone pair")
                logger.info("negative_code_tokens: {}".format([x.replace('\u0120', '_') for x in example.negative_code_tokens]))
                logger.info("negative_code_ids: {}".format(' '.join(map(str, example.negative_code_ids))))
                logger.info("negative_position_idx: {}".format(example.negative_position_idx))
                logger.info("negative_dfg_to_code: {}".format(' '.join(map(str, example.negative_dfg_to_code))))
                logger.info("negative_dfg_to_dfg: {}".format(' '.join(map(str, example.negative_dfg_to_dfg))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        ### anchor code pair
        # calculate graph-guided masked function
        anchor_attn_mask = np.zeros((self.args.code_length + self.args.data_flow_length,
                                     self.args.code_length + self.args.data_flow_length), dtype=np.bool_)
        # calculate begin index of node and max length of input
        anchor_node_index = sum([i > 1 for i in self.examples[item].anchor_position_idx])
        anchor_max_length = sum([i != 1 for i in self.examples[item].anchor_position_idx])
        # sequence can attend to sequence
        anchor_attn_mask[:anchor_node_index, :anchor_node_index] = True
        # special tokens attend to all tokensq
        for idx, i in enumerate(self.examples[item].anchor_code_ids):
            if i in [0, 2]:
                anchor_attn_mask[idx, :anchor_max_length] = True
        # nodes attend to code tokens that are identified from
        for idx, (a, b) in enumerate(self.examples[item].anchor_dfg_to_code):
            if a < anchor_node_index and b < anchor_node_index:
                anchor_attn_mask[idx + anchor_node_index, a:b] = True
                anchor_attn_mask[a:b, idx + anchor_node_index] = True
        # nodes attend to adjacent nodes
        for idx, nodes in enumerate(self.examples[item].anchor_dfg_to_dfg):
            for a in nodes:
                if a + anchor_node_index < len(self.examples[item].anchor_position_idx):
                    anchor_attn_mask[idx + anchor_node_index, a + anchor_node_index] = True

        ### positive code clone pair attn_mask
        # calculate graph-guided masked function
        positive_attn_mask = np.zeros((self.args.code_length + self.args.data_flow_length,
                                       self.args.code_length + self.args.data_flow_length), dtype=np.bool_)
        # calculate begin index of node and max length of input
        positive_node_index = sum([i > 1 for i in self.examples[item].positive_position_idx])
        positive_max_length = sum([i != 1 for i in self.examples[item].positive_position_idx])
        # sequence can attend to sequence
        positive_attn_mask[:positive_node_index, :positive_node_index] = True
        # special tokens attend to all tokens
        for idx, i in enumerate(self.examples[item].positive_code_ids):
            if i in [0, 2]:
                positive_attn_mask[idx, :positive_max_length] = True
        # nodes attend to code tokens that are identified from
        for idx, (a, b) in enumerate(self.examples[item].positive_dfg_to_code):
            if a < positive_node_index and b < positive_node_index:
                positive_attn_mask[idx + positive_node_index, a:b] = True
                positive_attn_mask[a:b, idx + positive_node_index] = True
        # nodes attend to adjacent nodes
        for idx, nodes in enumerate(self.examples[item].positive_dfg_to_dfg):
            for a in nodes:
                if a + positive_node_index < len(self.examples[item].positive_position_idx):
                    positive_attn_mask[idx + positive_node_index, a + positive_node_index] = True

        ### negative code clone pair attn_mask
        # calculate graph-guided masked function
        negative_attn_mask = np.zeros((self.args.code_length + self.args.data_flow_length,
                                       self.args.code_length + self.args.data_flow_length), dtype=np.bool_)
        # calculate begin index of node and max length of input
        negative_node_index = sum([i > 1 for i in self.examples[item].negative_position_idx])
        negative_max_length = sum([i != 1 for i in self.examples[item].negative_position_idx])
        # sequence can attend to sequence
        negative_attn_mask[:negative_node_index, :negative_node_index] = True
        # special tokens attend to all tokens
        for idx, i in enumerate(self.examples[item].negative_code_ids):
            if i in [0, 2]:
                negative_attn_mask[idx, :negative_max_length] = True
        # nodes attend to code tokens that are identified from
        for idx, (a, b) in enumerate(self.examples[item].negative_dfg_to_code):
            if a < negative_node_index and b < negative_node_index:
                negative_attn_mask[idx + negative_node_index, a:b] = True
                negative_attn_mask[a:b, idx + negative_node_index] = True
        # nodes attend to adjacent nodes
        for idx, nodes in enumerate(self.examples[item].negative_dfg_to_dfg):
            for a in nodes:
                if a + negative_node_index < len(self.examples[item].negative_position_idx):
                    negative_attn_mask[idx + negative_node_index, a + negative_node_index] = True

        return (torch.tensor(self.examples[item].anchor_code_ids),
                torch.tensor(anchor_attn_mask),
                torch.tensor(self.examples[item].anchor_position_idx),
                torch.tensor(self.examples[item].positive_code_ids),
                torch.tensor(positive_attn_mask),
                torch.tensor(self.examples[item].positive_position_idx),
                torch.tensor(self.examples[item].negative_code_ids),
                torch.tensor(negative_attn_mask),
                torch.tensor(self.examples[item].negative_position_idx)
                )


def get_ground_truth_pair(args, file_path):
    prefix = file_path.split('/')[-1][:-5]
    cache_file_ground_truth_pair_list = args.output_dir + '/' + prefix + '_ground_truth_pair.pkl'
    cache_file_pair_idx_list = args.output_dir + '/' + prefix + '_pair_idx_dict.pkl'
    cache_file_codebase_idx_list = args.output_dir + '/' + prefix + '_codebase_list.pkl'
    cache_file_codebase_choose_list = args.output_dir + '/' + prefix + '_codebase_choose_list.pkl'
    count = 0
    ground_truth_pairs = []
    pair_idx_lists = []
    codebase_idx_lists = []
    codebase_choose_lists = []
    if os.path.exists(cache_file_ground_truth_pair_list) and os.path.exists(cache_file_pair_idx_list) \
            and os.path.exists(cache_file_codebase_idx_list) and os.path.exists(cache_file_codebase_choose_list):
        ground_truth_pairs = pickle.load(open(cache_file_ground_truth_pair_list, 'rb'))
        pair_idx_lists = pickle.load(open(cache_file_pair_idx_list, 'rb'))
        codebase_idx_lists = pickle.load(open(cache_file_codebase_idx_list, 'rb'))
        codebase_choose_lists = pickle.load(open(cache_file_codebase_choose_list, 'rb'))
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                js = json.loads(line)
                clone_pair = {}
                clone_pair['func_1_id'] = js['func_1_id']
                clone_pair['func_1_name'] = js['func_1_name']
                clone_pair['func_2_id'] = js['func_2_id']
                clone_pair['func_2_name'] = js['func_2_name']
                ground_truth_pairs.append(clone_pair)

                pair_idx = []
                pair_idx.append(js['func_1_id'])
                pair_idx.append(js['func_2_id'])
                pair_idx_lists.append(pair_idx)

                if js['func_2_id'] not in codebase_idx_lists:
                    codebase_idx_lists.append(js['func_2_id'])
                    codebase_choose_lists.append(idx)
        pickle.dump(ground_truth_pairs, open(cache_file_ground_truth_pair_list, 'wb'))
        pickle.dump(pair_idx_lists, open(cache_file_pair_idx_list, 'wb'))
        pickle.dump(codebase_idx_lists, open(cache_file_codebase_idx_list, 'wb'))
        pickle.dump(codebase_choose_lists, open(cache_file_codebase_choose_list, 'wb'))
    return ground_truth_pairs, pair_idx_lists, codebase_idx_lists, codebase_choose_lists


def set_seed(seed=1234):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, model, tokenizer, pool, second_stage_model):
    """ Train the model """
    # get training dataset
    train_dataset = CodeDataset(tokenizer, args, args.train_data_file, pool)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4)

    # get optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * args.num_train_epochs)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size // args.n_gpu)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader) * args.num_train_epochs)

    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()

    model.train()
    tr_num, tr_loss, best_mrr = 0, 0, 0
    for epoch_idx in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            # get inputs
            anchor_code_inputs = batch[0].to(args.device)
            anchor_attn_mask = batch[1].to(args.device)
            anchor_position_idx = batch[2].to(args.device)

            positive_code_inputs = batch[3].to(args.device)
            positive_attn_mask = batch[4].to(args.device)
            positive_position_idx = batch[5].to(args.device)

            negative_code_inputs = batch[6].to(args.device)
            negative_attn_mask = batch[7].to(args.device)
            negative_position_idx = batch[8].to(args.device)

            hash_scale = math.pow((1.0 * step + 1.0), 0.5)
            anchor_code_vec = model(code_inputs=anchor_code_inputs, attn_mask=anchor_attn_mask, position_idx=anchor_position_idx, hash_scale=hash_scale, retrieval_model=False)
            positive_code_vec = model(code_inputs=positive_code_inputs, attn_mask=positive_attn_mask, position_idx=positive_position_idx, hash_scale=hash_scale, retrieval_model=False)
            negative_code_vec = model(code_inputs=negative_code_inputs, attn_mask=negative_attn_mask, position_idx=negative_position_idx, hash_scale=hash_scale, retrieval_model=False)

            positive_code_similarity = F.cosine_similarity(anchor_code_vec[1], positive_code_vec[1], dim=1).clamp(min=-1, max=1)
            negative_code_similarity = F.cosine_similarity(anchor_code_vec[1], negative_code_vec[1], dim=1).clamp(min=-1, max=1)
            loss = (0.5 - positive_code_similarity + negative_code_similarity).clamp(min=1e-6).mean()

            tr_loss += loss.item()
            tr_num += 1
            if (step+1) % 100 == 0:
                logger.info("epoch: {}, step: {}, loss: {}".format(epoch_idx, step+1, round(tr_loss/tr_num, 5)))
                tr_loss = 0
                tr_num = 0

            # backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        results = evaluate(args, model, tokenizer, args.valid_data_file, pool, hash_scale, second_stage_model, eval_when_training=True)
        for key, value in results.items():
            logger.info("epoch: %s, %s = %s", epoch_idx, key, round(value, 4))

        # save best model
        if results['valid_mrr'] > best_mrr:
            best_mrr = results['valid_mrr']
            logger.info("  " + "*" * 20)
            logger.info("  Epoch: %s, Best mrr:%s", epoch_idx, round(best_mrr, 4))
            logger.info("  " + "*" * 20)

            checkpoint_prefix = 'checkpoint-best-mrr'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model
            output_dir = os.path.join(output_dir, '{}'.format('model.bin'))
            torch.save(model_to_save.state_dict(), output_dir)
            logger.info("Saving model checkpoint to %s", output_dir)

def get_martix_code_similarity(v1, v2):
    num = np.dot(v1, np.array(v2).T)
    denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1)  # 求模长的乘积
    res = num / denom
    res[np.isneginf(res)] = 0
    return 0.5 + 0.5 * res

def evaluate(args, model, tokenizer, file_name, pool, hash_scale, second_stage_model, eval_when_training=False):
    """ Evaluate the model """
    # get test dataset
    query_code_dataset = CodeDataset(tokenizer, args, file_name, pool)
    query_code_sampler = SequentialSampler(query_code_dataset)
    query_code_dataloader = DataLoader(query_code_dataset, sampler=query_code_sampler, batch_size=args.valid_batch_size, num_workers=4)

    codebase_dataset = CodeDataset(tokenizer, args, file_name, pool)
    codebase_sampler = SequentialSampler(codebase_dataset)
    codebase_dataloader = DataLoader(codebase_dataset, sampler=codebase_sampler, batch_size=args.valid_batch_size, num_workers=4)

    # get ground truth
    ground_truth_pairs, pair_idx_lists, codebase_idx_lists, codebase_choose_lists = get_ground_truth_pair(args, file_name)
    codebase_index_list = list(range(len(codebase_choose_lists)))

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    query_source_code_snippets = []
    codebase_source_code_snippets = []
    with open(file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            line = line.strip()
            js = json.loads(line)
            query_source_code_snippets.append(js['func_1_source_code'].strip())
            codebase_source_code_snippets.append(js['func_2_source_code'].strip())

    codebase_source_code_snippets = np.array(codebase_source_code_snippets)[codebase_choose_lists].tolist()

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num queries = %d", len(query_code_dataset))
    logger.info("  Num codes = %d", len(codebase_choose_lists))
    logger.info("  Batch size = %d", args.valid_batch_size)

    model.eval()
    codebase_dense_vecs = []
    codebase_hash_vecs = []

    hash_scale = math.pow((1.0 * hash_scale + 1.0), 0.5)
    for codebase_batch in codebase_dataloader:
        positive_code_inputs = codebase_batch[3].to(args.device)
        positive_attn_mask = codebase_batch[4].to(args.device)
        positive_position_idx = codebase_batch[5].to(args.device)
        with torch.no_grad():
            # hash vector for hashing fast search
            codebase_hash_like_vec_batch, codebase_hash_vec_batch = model(code_inputs=positive_code_inputs, attn_mask=positive_attn_mask, position_idx=positive_position_idx, hash_scale=hash_scale, retrieval_model=True)
            codebase_hash_vecs.append(codebase_hash_vec_batch.cpu().numpy())

            # dense vector for cosine similarity
            codebase_dense_vec_batch = second_stage_model(code_inputs=positive_code_inputs, attn_mask=positive_attn_mask, position_idx=positive_position_idx)
            codebase_dense_vecs.append(codebase_dense_vec_batch.cpu().numpy())

    codebase_dense_vecs = np.concatenate(codebase_dense_vecs, 0)[codebase_choose_lists]

    codebase_hash_vecs = np.concatenate(codebase_hash_vecs, 0)[codebase_choose_lists]
    codebase_hash_vecs = (codebase_hash_vecs > 0).astype(np.uint8)

    index = faiss.IndexBinaryFlat(len(codebase_hash_vecs[0]) * 8)
    index.add(codebase_hash_vecs)  # 给codebase_hash index

    query_code__code_inputs = []
    query_code__attn_mask = []
    query_code__position_idx = []
    for query_batch in query_code_dataloader:
        anchor_query_code_inputs = query_batch[0].to(args.device)
        anchor_query_attn_mask = query_batch[1].to(args.device)
        anchor_query_position_idx = query_batch[2].to(args.device)
        with torch.no_grad():
            query_code__code_inputs.append(anchor_query_code_inputs)
            query_code__attn_mask.append(anchor_query_attn_mask)
            query_code__position_idx.append(anchor_query_position_idx)

    query_code__code_inputs = torch.cat(query_code__code_inputs, 0)
    query_code__attn_mask = torch.cat(query_code__attn_mask, 0)
    query_code__position_idx = torch.cat(query_code__position_idx, 0)

    query_code_inputs = []
    for i in range(len(query_code__code_inputs)):
        query_code_dict = {}
        query_code_dict['anchor_query_code_inputs'] = query_code__code_inputs[i]
        query_code_dict['anchor_query_attn_mask'] = query_code__attn_mask[i]
        query_code_dict['anchor_query_position_idx'] = query_code__position_idx[i]

        query_code_inputs.append(query_code_dict)

    #### record time
    embedding_generation_time_cost = 0
    retrieval_stage_time_cost = 0
    rerank_stage_time_cost = 0

    test_start = datetime.now()
    sort_ids = []
    count = 0
    for query_code in query_code_inputs:
        test_start_per_query = datetime.now()

        query_code__code_inputs = query_code['anchor_query_code_inputs']
        query_code__attn_mask = query_code['anchor_query_attn_mask']
        query_code__position_idx = query_code['anchor_query_position_idx']
        with torch.no_grad():
            query_code_hash_like_vec, query_code_hash_vec = model(code_inputs=query_code__code_inputs.unsqueeze(0), attn_mask=query_code__attn_mask.unsqueeze(0), position_idx=query_code__position_idx.unsqueeze(0), hash_scale=hash_scale, retrieval_model=True)
            query_code_hash_vec = (query_code_hash_vec.cpu().numpy() > 0).astype(np.uint8)

            embedding_generation_test_end_per_query = datetime.now()
            D, I = index.search(query_code_hash_vec, 100)
            canndidate_code_dense_vecs = codebase_dense_vecs[I[0].tolist()]

            retrieval_stage_test_end_per_query = datetime.now()
            scores = get_martix_code_similarity(query_code_hash_like_vec.cpu().numpy(), canndidate_code_dense_vecs)

            sort_id = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:, ::-1]
            sort_ids.append(np.array(codebase_idx_lists)[I[0]][sort_id[0]])

            rerank_stage_test_end_per_query = datetime.now()

            embedding_generation_time_cost += (embedding_generation_test_end_per_query - test_start_per_query).total_seconds()  # embedding generation time cost
            retrieval_stage_time_cost += (retrieval_stage_test_end_per_query - embedding_generation_test_end_per_query).total_seconds()  # retrieval stage time cost
            rerank_stage_time_cost += (rerank_stage_test_end_per_query - retrieval_stage_test_end_per_query).total_seconds()  # rerank stage time cost
            print("--------------------------------------------")
            print(f"input index: {count}, input query")
            print(query_source_code_snippets[count])
            print(f"Two-Stage hash-based search result")
            for i in range(3):
                print(f"searched result: {i}")
                print(codebase_source_code_snippets[I[0][sort_id[0]][i]])
            print("--------------------------------------------")
            count += 1

    logger.info("--------------------------------------------")
    logger.info(f"Time elapsed for {len(sort_ids)} times code search = {datetime.now() - test_start}")
    logger.info(f"Average time elapsed for per code search = {(datetime.now() - test_start) / len(sort_ids)}")
    logger.info("--------------------------------------------")
    logger.info(f"Embedding generation of R2CC time cost: {embedding_generation_time_cost}")
    logger.info(f"Retrieval stage of R2CC time cost: {retrieval_stage_time_cost}")
    logger.info(f"Rerank stage of R2CC time cost: {rerank_stage_time_cost}")
    logger.info(f"Total time of three stage cost: {embedding_generation_time_cost + retrieval_stage_time_cost + rerank_stage_time_cost}")

    ndarray_codebase_idx_lists = np.array(codebase_idx_lists)

    mrrs = []
    for i in range(len(sort_ids)):
        if pair_idx_lists[i][1] in sort_ids[i]:
            rank = sort_ids[i].tolist().index(pair_idx_lists[i][1])
            if rank == 0:
                mrrs.append(1.0)
            else:
                mrrs.append(1.0 / rank)
        else:
            mrrs.append(0)
    logger.info(f"len(mrrs): {len(mrrs)}")
    result = {
        "mrr": float(np.mean(mrrs))
    }

    success = {1: 0, 5: 0, 10: 0, 100: 0}
    for i in range(len(sort_ids)):
        if pair_idx_lists[i][1] in sort_ids[i]:
            rank = sort_ids[i].tolist().index(pair_idx_lists[i][1])
        else:
            rank = 101
        for k in success.keys():
            if rank <= k:
                success[k] += 1
    logger.info(f"Top@k: {success}")

    for k in success.keys():
        success[k] = success[k] / len(sort_ids)
    logger.info(f"Percentage Top@k: {success}")

    return result


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--train_data_file", default=None, type=str, required=False,
                        help="The input training data file (a json file).")
    parser.add_argument("--valid_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the MRR(a jsonl file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--lang", default=None, type=str,
                        help="language.")

    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.")
    parser.add_argument("--data_flow_length", default=64, type=int,
                        help="Optional Data Flow input sequence length after tokenization.")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_valid", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--valid_batch_size", default=4, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    pool = multiprocessing.Pool(cpu_cont)

    # print arguments
    args = parser.parse_args()

    # set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s", device, args.n_gpu)

    # Set seed
    set_seed(args.seed)

    # build model
    config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    pre_train_model = RobertaModel.from_pretrained(args.model_name_or_path)
    model = Model(pre_train_model)
    logger.info("Training/evaluation parameters %s", args)
    model.to(args.device)

    # build second stage model for specific dense vector search
    second_stage_model = Second_Stage_Model(pre_train_model)
    second_stage_model_path = '/mnt/silver/huangxin/model/fast-code-search/graphcodebert-code2code-train_all_true_055not045-valid_unique_test_unique/saved_models/bigclonebench/checkpoint-best-mrr/model.bin'
    second_stage_model.load_state_dict(torch.load(second_stage_model_path), strict=False)
    second_stage_model.to(args.device)
    for name, parameter in second_stage_model.named_parameters():
        parameter.requires_grad = False

    # Training
    if args.do_train:
        train(args, model, tokenizer, pool, second_stage_model)

    # Evaluation
    results = {}
    if args.do_valid:
        checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir), strict=False)
        model.to(args.device)
        result = evaluate(args, model, tokenizer, args.valid_data_file, pool, 38000, second_stage_model)
        logger.info("***** Valid results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key], 4)))

    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir), strict=False)
        model.to(args.device),
        result = evaluate(args, model, tokenizer, args.test_data_file, pool, 38000, second_stage_model)
        logger.info("***** Test results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key], 4)))
    return results


if __name__ == "__main__":
    main()
