import time
import numpy as np 
import codecs
import argparse
from dataset.loader import load_data
from model.Model import Transformer
from trainer import Trainer
from evaluator import Evaluator
import model.Constants as Constants
from logger import initialize_exp
from model.LM import LanguageModel
import torch
import os


def get_parser():
    parser = argparse.ArgumentParser(description='Text Simplification')

    parser.add_argument("--simp_train_path", type=str, default="")
    parser.add_argument("--autoencoder_path", type=str, default="")
    parser.add_argument("--simp_dev_path", type=str, default="")
    parser.add_argument("--comp_train_path", type=str, default="")
    parser.add_argument("--comp_dev_path", type=str, default="")

    parser.add_argument("--para_dev_path", type=str, default="")
    parser.add_argument("--frc_path", type=str, default="")
    parser.add_argument("--para_test_path", type=str, default="")
    parser.add_argument("--para_train_path", type=str, default="")
    parser.add_argument("--supervised_rate", type=int, default=0)
    parser.add_argument("--vocab_path", type=str, default="")
    parser.add_argument("--us_pretrain_embedding", type=int, default=1)
    parser.add_argument("--embedding_path", type=str, default="")
    parser.add_argument("--comp_frequent_list", type=str, default="")
    parser.add_argument("--simp_frequent_list", type=str, default="")
    parser.add_argument("--comp_ppdb_rules", type=str, default="")
    parser.add_argument("--simp_ppdb_rules", type=str, default="")
    parser.add_argument("--dump_path", type=str, default="")
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--stoplist_path", type=str, default="")
    parser.add_argument("--use_pretrained_model", type=int, default=0)
    parser.add_argument("--otf_autoencoding", type=int, default=0)
    parser.add_argument("--otf_back_translation", type=int, default=0)


    # transformer parameters
    parser.add_argument("--emb_dim", type=int, default=512,
                        help="Embedding layer size")
    parser.add_argument("--n_enc_layers", type=int, default=4,
                        help="Number of layers in the encoders")
    parser.add_argument("--n_dec_layers", type=int, default=4,
                        help="Number of layers in the decoders")

    parser.add_argument("--dropout", type=float, default=0,
                        help="Dropout")
    parser.add_argument("--d_inner", type=int, default=2048,
                        help="Transformer fully-connected hidden dim size")
    parser.add_argument("--n_head", type=int, default=8,
                        help="encoder_attention_heads")
    parser.add_argument("--d_model", type=int, default=512,
                        help="hidden size of transformer, must equal with embedding dim")
    parser.add_argument("--d_k", type=int, default=8,
                        help="size of keys")
    parser.add_argument("--d_v", type=int, default=8,
                        help="size of value")
    parser.add_argument("--len_max_seq", type=int, default=100,
                        help="size of value")

    parser.add_argument("--share_encdec_emb", type=int, default=0,
                        help="Share encoder embeddings / decoder embeddings")
    parser.add_argument("--share_decpro_emb", type=int, default=0,
                        help="Share decoder embeddings / decoder output projection")
    parser.add_argument("--share_output_emb", type=int, default=0,
                        help="Share decoder output embeddings")

    parser.add_argument("--share_enc", type=int, default=0,
                        help="Number of layers to share in the encoders")
    parser.add_argument("--share_dec", type=int, default=0,
                        help="Number of layers to share in the decoders")

    # encoder input perturbation
    parser.add_argument("--word_shuffle", type=float, default=0,
                        help="Randomly shuffle input words (0 to disable)")
    parser.add_argument("--shuffle_mode", type=str, default="")
    parser.add_argument("--drop_type", type=str, default="")
    parser.add_argument("--word_replace", type=float, default=0,
                        help="Randomly replace input words (0 to disable)")
    parser.add_argument("--word_dropout", type=float, default=0,
                        help="Randomly dropout input words (0 to disable)")
    parser.add_argument("--word_blank", type=float, default=0,
                        help="Randomly blank input words (0 to disable)")
    parser.add_argument("--syn_denosing", type=float, default=0,
                        help="Use syntactic denosing")

    # training steps
    parser.add_argument("--otf_sample", type=float, default=-1,
                        help="Temperature for sampling back-translations (-1 for greedy decoding)")
    parser.add_argument("--otf_backprop_temperature", type=float, default=-1,
                        help="Back-propagate through the encoder (-1 to disable, temperature otherwise)")
    parser.add_argument("--otf_sync_params_every", type=int, default=1000, metavar="N",
                        help="Number of updates between synchronizing params")
    parser.add_argument("--otf_num_processes", type=int, default=30, metavar="N",
                        help="Number of processes to use for OTF generation")
    parser.add_argument("--otf_update_enc", type=int, default=True,
                        help="Update the encoder during back-translation training")
    parser.add_argument("--otf_update_dec", type=int, default=True,
                        help="Update the decoder during back-translation training")
    parser.add_argument("--stopping_criterion", type=str, default=None)

    # training parameters
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--use_multi_process", type=int, default=0,
                        help="use_multi_process")

    parser.add_argument("--lambda_xe_mono", type=int, default=1,
                        help="Cross-entropy reconstruction coefficient (autoencoding)")
    parser.add_argument("--lambda_xe_para", type=str, default="0",
                        help="Cross-entropy reconstruction coefficient (parallel data)")
    parser.add_argument("--lambda_xe_back", type=str, default="0",
                        help="Cross-entropy reconstruction coefficient (back-parallel data)")
    parser.add_argument("--lambda_xe_otfd", type=str, default="0",
                        help="Cross-entropy reconstruction coefficient (on-the-fly back-translation parallel data)")
    parser.add_argument("--lambda_xe_otfa", type=str, default="0",
                        help="Cross-entropy reconstruction coefficient (on-the-fly back-translation autoencoding data)")

    parser.add_argument("--epoch_size", type=int, default=100000,
                        help="Epoch size / evaluation frequency")
    parser.add_argument("--max_epoch", type=int, default=100000,
                        help="Maximum epoch size")
    parser.add_argument("--pretrain_autoencoder", type=int, default=0)
    parser.add_argument("--rl_finetune", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--delta", type=float, default=0.5)
    parser.add_argument("--use_lm", type=int, default=0)
    parser.add_argument("--lm_path", type=str, default="")
    parser.add_argument("--additive", type=int, default=1)
    parser.add_argument('--simp_drop', type=int, default=0)
    parser.add_argument('--use_check', type=int, default=0)

    # freeze network parameters
    parser.add_argument("--freeze_enc_emb", type=int, default=0,
                        help="Freeze encoder embeddings")
    parser.add_argument("--freeze_dec_emb", type=int, default=0,
                        help="Freeze decoder embeddings")
    # evaluation
    parser.add_argument("--eval_only", type=int, default=0,
                        help="Only run evaluations")
    parser.add_argument("--beam_size", type=int, default=0,
                        help="Beam width (<= 0 means greedy)")
    return parser


def main(params):

    def anneal_function(step, k, x0):
        return float(params.gamma / (1+np.exp(-k*(step-x0))))

    logger = initialize_exp(params)
    data = load_data(params)
    params.n_src_vocab = len(data['index2word'])
    params.n_tgt_vocab = len(data['index2word'])
    model = Transformer(params).to(Constants.device)

    if params.use_pretrained_model:
        logger.info("loading pretrained model")
        path = os.path.join(params.dump_path, '%s.pth' % params.name)
        model_data = torch.load(path)
        model = model_data['model'].to(Constants.device)

    elif params.pretrain_autoencoder == 0:
        logger.info("loading pretrained autoencoders")
        path = os.path.join(params.dump_path, 'autoencoder.pth')
        model_data = torch.load(path)
        model = model_data['model'].to(Constants.device)

    if params.use_lm:
        logger.info("loading pretrained language model")
        path = params.lm_path
        lm = torch.load(path).to(Constants.device)
    else:
        lm = None

    trainer = Trainer(model, lm, data, params, logger)
    if params.use_check:
        trainer.reload_checkpoint()
    evaluator = Evaluator(trainer.model, lm, data, params)


    logger.info("==================== Eval at Random parameters =====================")
    #scores = evaluator.eval_all(use_pointer=False)

    logger.info(" ====================== Pretraing Embedding... ====================")

    if params.pretrain_autoencoder > 0:
        for i in range(params.pretrain_autoencoder):
            trainer.enc_dec_step('simp', 'simp')
            trainer.enc_dec_step('comp', 'comp')
            if i % 5000 == 0:
                simp_loss, comp_loss = trainer.print_stats(pretrain=True)
                # score = evaluator.eval_all(use_pointer=False)
                
        logger.info("saving model")
        trainer.save_model(params.name)
        return

    logger.info("==================== Eval at AutoEncoder Only ====================")
    scores = evaluator.eval_all(use_pointer=False)

    trainer.start_back_translation()
    for ep in range(params.max_epoch):
        logger.info(" ======================== Start Epoch %i ======================" % ep)
        trainer.n_sentences = 0

        while trainer.n_sentences < params.epoch_size:
            trainer.start_time = time.time()

            if params.otf_autoencoding:
                mono_xe = 1
                if mono_xe > 0:
                    trainer.enc_dec_step('simp', 'simp', xe=mono_xe)
                    trainer.enc_dec_step('comp', 'comp', xe=mono_xe)

            if params.supervised_rate > 0:
                trainer.enc_dec_step('comp', 'simp', xe=1, back=False)

            if params.otf_back_translation:
                if trainer.n_iter % params.otf_sync_params_every == 0:
                    logger.info("Synchronize the model parameters")
                    trainer.otf_sync_params()

                if not getattr(params, 'started_otf_batch_gen', False):
                    otf_iterator = trainer.otf_bt_gen_async()
                    params.started_otf_batch_gen = True

                if trainer.n_iter % params.otf_sync_params_every == 0:
                    trainer.otf_sync_params()

                if params.supervised_rate == 0:
                    otf_gamma = anneal_function(trainer.n_iter, k=0.00075, x0=10000)
                else:
                    otf_gamma = params.gamma

                before_gen = time.time()
                batches = next(otf_iterator)
                trainer.gen_time += time.time() - before_gen

                for batch in batches:
                    trainer.otf_bt(batch, lambda_xe=params.lambda_xe_otfd, gamma=otf_gamma)

            trainer.iter()

        logger.info("*********** Evaluating ***********")
        scores = evaluator.eval_all(use_pointer=False)
        sari = float(scores['sari'])
        # trainer.model_scheduler.step(sari)
        is_end = trainer.end_epoch(scores)

        if is_end:
            break


if __name__ == '__main__':
    parser = get_parser()
    params = parser.parse_args()
    main(params)

