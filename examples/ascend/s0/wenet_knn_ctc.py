from __future__ import print_function

import argparse
import copy
import logging
import os
import sys
import time
from datetime import datetime

import torch
import yaml
from torch.utils.data import DataLoader

from wenet.dataset.dataset import Dataset
from wenet.utils.config import override_config
from wenet.utils.init_model import init_model
from wenet.utils.init_tokenizer import init_tokenizer
from wenet.utils.context_graph import ContextGraph
from wenet.utils.ctc_utils import get_blank_id
from wenet.utils.common import TORCH_NPU_AVAILABLE  # noqa just ensure to check torch-npu

from wenet_knn_ctc_model import KNNSaver_for_ctc, KNNWrapper_for_ctc, KEY_TYPE, DIST


def get_args():
    parser = argparse.ArgumentParser(description='recognize with your model')
    
    # ---------------------------------------------------------
    # normal recogize.py arguments
    # ---------------------------------------------------------
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--test_data', required=True, help='test data file')
    parser.add_argument('--data_type',
                        default='raw',
                        choices=['raw', 'shard'],
                        help='train and cv data type')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--device',
                        type=str,
                        default="cpu",
                        choices=["cpu", "npu", "cuda"],
                        help='accelerator to use')
    parser.add_argument('--dtype',
                        type=str,
                        default='fp32',
                        choices=['fp16', 'fp32', 'bf16'],
                        help='model\'s dtype')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--beam_size',
                        type=int,
                        default=10,
                        help='beam size for search')
    parser.add_argument('--length_penalty',
                        type=float,
                        default=0.0,
                        help='length penalty')
    parser.add_argument('--blank_penalty',
                        type=float,
                        default=0.0,
                        help='blank penalty')
    parser.add_argument('--result_dir', required=True, help='asr result file')
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='asr result file')
    parser.add_argument('--modes',
                        nargs='+',
                        help="""decoding mode, support the following:
                             attention
                             ctc_greedy_search
                             ctc_prefix_beam_search
                             attention_rescoring
                             rnnt_greedy_search
                             rnnt_beam_search
                             rnnt_beam_attn_rescoring
                             ctc_beam_td_attn_rescoring
                             hlg_onebest
                             hlg_rescore
                             paraformer_greedy_search
                             paraformer_beam_search""")
    parser.add_argument('--search_ctc_weight',
                        type=float,
                        default=1.0,
                        help='ctc weight for nbest generation')
    parser.add_argument('--search_transducer_weight',
                        type=float,
                        default=0.0,
                        help='transducer weight for nbest generation')
    parser.add_argument('--ctc_weight',
                        type=float,
                        default=0.0,
                        help='ctc weight for rescoring weight in \
                                  attention rescoring decode mode \
                              ctc weight for rescoring weight in \
                                  transducer attention rescore decode mode')

    parser.add_argument('--transducer_weight',
                        type=float,
                        default=0.0,
                        help='transducer weight for rescoring weight in '
                        'transducer attention rescore mode')
    parser.add_argument('--attn_weight',
                        type=float,
                        default=0.0,
                        help='attention weight for rescoring weight in '
                        'transducer attention rescore mode')
    parser.add_argument('--decoding_chunk_size',
                        type=int,
                        default=-1,
                        help='''decoding chunk size,
                                <0: for decoding, use full chunk.
                                >0: for decoding, use fixed chunk size as set.
                                0: used for training, it's prohibited here''')
    parser.add_argument('--num_decoding_left_chunks',
                        type=int,
                        default=-1,
                        help='number of left chunks for decoding')
    parser.add_argument('--simulate_streaming',
                        action='store_true',
                        help='simulate streaming inference')
    parser.add_argument('--reverse_weight',
                        type=float,
                        default=0.0,
                        help='''right to left weight for attention rescoring
                                decode mode''')
    parser.add_argument('--override_config',
                        action='append',
                        default=[],
                        help="override yaml config")

    parser.add_argument('--word',
                        default='',
                        type=str,
                        help='word file, only used for hlg decode')
    parser.add_argument('--hlg',
                        default='',
                        type=str,
                        help='hlg file, only used for hlg decode')
    parser.add_argument('--lm_scale',
                        type=float,
                        default=0.0,
                        help='lm scale for hlg attention rescore decode')
    parser.add_argument('--decoder_scale',
                        type=float,
                        default=0.0,
                        help='lm scale for hlg attention rescore decode')
    parser.add_argument('--r_decoder_scale',
                        type=float,
                        default=0.0,
                        help='lm scale for hlg attention rescore decode')

    parser.add_argument(
        '--context_bias_mode',
        type=str,
        default='',
        help='''Context bias mode, selectable from the following
                                option: decoding-graph, deep-biasing''')
    parser.add_argument('--context_list_path',
                        type=str,
                        default='',
                        help='Context list path')
    parser.add_argument('--context_graph_score',
                        type=float,
                        default=0.0,
                        help='''The higher the score, the greater the degree of
                                bias using decoding-graph for biasing''')

    parser.add_argument('--use_lora',
                        type=bool,
                        default=False,
                        help='''Whether to use lora for biasing''')
    parser.add_argument("--lora_ckpt_path",
                        default=None,
                        type=str,
                        help="lora checkpoint path.")

    # ---------------------------------------------------------
    # external knn arguments
    # ---------------------------------------------------------
    parser.add_argument('--knn', action='store_true')
    parser.add_argument('--knn_gpu', action='store_true')
    parser.add_argument('--dstore_size', type=int, nargs='+', default=None)
    parser.add_argument('--knn_keytype', type=KEY_TYPE.from_string, default=KEY_TYPE.last_ffn_input)
    parser.add_argument('--save_knnlm_dstore', action='store_true')
    parser.add_argument('--dstore_dir', type=str, nargs='+',default=None)
    parser.add_argument('--knn_sim_func', type=DIST.from_string, default=DIST.l2)
    parser.add_argument('--lmbda', type=float, default=0.25)
    parser.add_argument('--k', type=int, default=1024)
    parser.add_argument('--n', type=int, default=300)
    parser.add_argument('--t', type=float, default=1.0)
    parser.add_argument('--knn_temp', type=float, default=1.0)
    parser.add_argument('--build_index', action='store_true')
    parser.add_argument('--ncentroids', type=int, default=1024)
    parser.add_argument('--code_size', type=int, default=64)
    parser.add_argument('--probe', type=int, default=32)
    parser.add_argument('--num_keys_to_add_at_a_time', type=int, default=1000000)
    parser.add_argument('--move_dstore_to_mem', action='store_true')
    parser.add_argument('--no_load_keys', action='store_true')
    parser.add_argument('--recompute_dists', action='store_true')
    parser.add_argument('--thr', type=float, default=0.0, help='pseudo label thr')
    parser.add_argument('--decode_skip_blank', action='store_true')
    parser.add_argument('--scale_lmbda', action='store_true')
    parser.add_argument('--scale_lmbda_temp', type=float, default=1.0)
    parser.add_argument('--use_null_mask', action='store_true')

    args = parser.parse_args()
    print(args)
    return args


def main():
    # step 1, parser
    # ------------------------------
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')

    if args.gpu != -1:
        # remain the original usage of gpu
        args.device = "cuda"
    if "cuda" in args.device:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    if len(args.override_config) > 0:
        configs = override_config(configs, args.override_config)

    test_conf = copy.deepcopy(configs['dataset_conf'])

    test_conf['filter_conf']['max_length'] = 102400
    test_conf['filter_conf']['min_length'] = 0
    test_conf['filter_conf']['token_max_length'] = 102400
    test_conf['filter_conf']['token_min_length'] = 0
    test_conf['filter_conf']['max_output_input_ratio'] = 102400
    test_conf['filter_conf']['min_output_input_ratio'] = 0
    test_conf['speed_perturb'] = False
    test_conf['spec_aug'] = False
    test_conf['spec_sub'] = False
    test_conf['spec_trim'] = False
    test_conf['shuffle'] = False
    test_conf['sort'] = False
    test_conf['cycle'] = 1
    test_conf['list_shuffle'] = False
    if 'fbank_conf' in test_conf:
        test_conf['fbank_conf']['dither'] = 0.0
    elif 'mfcc_conf' in test_conf:
        test_conf['mfcc_conf']['dither'] = 0.0
    test_conf['batch_conf']['batch_type'] = "static"
    test_conf['batch_conf']['batch_size'] = args.batch_size

    # ------------------------------
    # step 2, init dataset and model
    # remember to switch dataset
    # when building index , use the training set
    # when test decoing method, use the testing set
    tokenizer = init_tokenizer(configs)
    test_dataset = Dataset(args.data_type,
                           args.test_data,
                           tokenizer,
                           test_conf,
                           partition=False)

    test_data_loader = DataLoader(test_dataset,
                                  batch_size=None,
                                  num_workers=args.num_workers)

    args.jit = False
    model, configs = init_model(args, configs)

    device = torch.device(args.device)
    model = model.to(device)
    model.eval()
    dtype = torch.float32
    if args.dtype == 'fp16':
        dtype = torch.float16
    elif args.dtype == 'bf16':
        dtype = torch.bfloat16
    logging.info("compute dtype is {}".format(dtype))

    context_graph = None
    if 'decoding-graph' in args.context_bias_mode:
        context_graph = ContextGraph(args.context_list_path,
                                     tokenizer.symbol_table,
                                     configs['tokenizer_conf']['bpe_path'],
                                     args.context_graph_score)

    _, blank_id = get_blank_id(configs, tokenizer.symbol_table)
    logging.info("blank_id is {}".format(blank_id))

    # ------------------------------
    # step 3, insert KNN
    dimension = configs["encoder_conf"]["output_size"]
    vocab_size = configs["output_dim"]
    knn_wrapper = None

    if args.build_index:
        knn_wrapper = KNNSaver_for_ctc(
            dstore_size=args.dstore_size[0],
            dstore_dir=args.dstore_dir[0],
            dimension=dimension,
            knn_keytype=args.knn_keytype,
            use_null_mask=args.use_null_mask,
        )
        knn_wrapper.register(model, args.thr, vocab_size)
    elif args.knn:
        # hardcode the language id ranges based on the symbol table
        symbol_table_path = symbol_table_path = configs['tokenizer_conf']['symbol_table_path']
        en_indices = list(range(4, 30))
        zh_indices = list(range(32, vocab_size))

        # check if we are in gated monolingual mode (2 directories) 
        # or in regular knn mode (1 directory)
        is_gated = len(args.dstore_dir) > 1 if args.dstore_dir else False

        if is_gated:
            logging.info("Initializing Gated Monolingual KNN Wrapper (Dual Datastores)")
        else:
            logging.info("Initializing Standard KNN Wrapper (Single Datastore)")

        knn_wrapper = KNNWrapper_for_ctc(
            dstore_size=args.dstore_size,
            dstore_dir=args.dstore_dir,
            dimension=dimension,
            knn_sim_func=args.knn_sim_func,
            knn_keytype=args.knn_keytype,
            no_load_keys=args.no_load_keys,
            move_dstore_to_mem=args.move_dstore_to_mem,
            knn_gpu=args.knn_gpu,
            recompute_dists=args.recompute_dists,
            k=args.k,
            n=args.n,
            t=args.t,
            lmbda=args.lmbda,
            knn_temp=args.knn_temp,
            probe=args.probe,
            decode_skip_blank=args.decode_skip_blank,
            scale_lmbda=args.scale_lmbda,
            scale_lmbda_temp=args.scale_lmbda_temp,
            zh_indices=zh_indices,
            en_indices=en_indices,
        )
        knn_wrapper.register(model)

    # ------------------------------
    # step 4: inference
    files = {}
    for mode in args.modes:
        dir_name = os.path.join(args.result_dir, mode)
        os.makedirs(dir_name, exist_ok=True)
        file_name = os.path.join(dir_name, 'text')
        files[mode] = open(file_name, 'w', encoding='utf-8')
    max_format_len = max([len(mode) for mode in args.modes])

    with torch.cuda.amp.autocast(enabled=True,
                                 dtype=dtype,
                                 cache_enabled=False):
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_data_loader):
                keys = batch["keys"]
                feats = batch["feats"].to(device)
                target = batch["target"].to(device)
                feats_lengths = batch["feats_lengths"].to(device)
                target_lengths = batch["target_lengths"].to(device)
                infos = {"tasks": batch["tasks"], "langs": batch["langs"]}
                results = model.decode(
                    args.modes,
                    feats,
                    feats_lengths,
                    args.beam_size,
                    decoding_chunk_size=args.decoding_chunk_size,
                    num_decoding_left_chunks=args.num_decoding_left_chunks,
                    ctc_weight=args.ctc_weight,
                    simulate_streaming=args.simulate_streaming,
                    reverse_weight=args.reverse_weight,
                    context_graph=context_graph,
                    blank_id=blank_id,
                    blank_penalty=args.blank_penalty,
                    length_penalty=args.length_penalty,
                    infos=infos,
                    knn_args=args,
                    knn_wrapper=knn_wrapper)
                for i, key in enumerate(keys):
                    for mode, hyps in results.items():
                        tokens = hyps[i].tokens
                        line = '{} {}'.format(key,
                                              tokenizer.detokenize(tokens)[0])
                        logging.info('{} {}'.format(mode.ljust(max_format_len),
                                                    line))
                        files[mode].write(line + '\n')
        for mode, f in files.items():
            f.close()

    if args.build_index:
        knn_wrapper.build_index()

    if args.build_index:
        dstore_index = knn_wrapper.show_dstore_index()
        current_path = os.path.dirname(args.result_dir)
        with open(os.path.join(current_path, "index_size"), "w") as f:
            f.write("the total datastore size is :" + str(dstore_index))


# ---------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
