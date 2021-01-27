import torch
from torch.nn import functional as F
import model.Constants as Constants
from multiprocessing_event_loop import MultiprocessingEventLoop
from model.Model import Transformer
import time
import os
from logging import getLogger
from nltk.translate.meteor_score import single_meteor_score
from metrics.FKGL import fkgl_score
from metrics.SARI import SARIsent
import numpy as np
from nltk.corpus import cmudict
from torch.distributions import Categorical
from torch.optim.lr_scheduler import ReduceLROnPlateau
from nltk.translate.bleu_score import corpus_bleu
from model.STS import STS_model
import codecs

logger = getLogger()


def merge_subword(subword_list):
    ret_sent = []
    align = []
    index = 0
    prev = ""
    for step, word in enumerate(subword_list):
        if "##" in word:
            prev += word.strip("##")
        else:
            if prev != "":
                ret_sent.append(prev)
                index += 1
            prev = word

        align.append(index)

    if prev != "":
        ret_sent.append(prev)

    return " ".join(ret_sent), align


class Trainer(MultiprocessingEventLoop):

    def __init__(self, model, lm, data, params, logger):
        super(Trainer, self).__init__(device_ids=tuple(range(params.otf_num_processes)))
        self.model = model
        self.lm = lm
        self.data = data
        self.params = params
        self.iterators = {}
        self.type_dict = {'comp': 0, 'simp': 1}
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001, betas=(0.5, 0.999))

        self.n_sentences = 0
        self.n_iter = 0
        self.gen_time = 0
        self.start_time = 0
        self.stopping_criterion = params.stopping_criterion
        self.best_stopping_criterion = -1
        self.relative_best = -1
        self.decrease_counts = 0
        self.decrease_counts_max = 30
        self.epoch = 0
        self.index2word = data['index2word']
        self.cmu_dict = cmudict.dict()
        self.rewards_simp = []
        self.rewards_comp = []
        self.sts_model = STS_model(params)

        self.stat = {
            'auto_encoder_simp_loss': [],
            'auto_encoder_comp_loss': [],
            'simp_comp_loss': [],
            'comp_simp_loss': [],
            'simp_comp_otf_loss': [],
            'comp_simp_otf_loss': []
        }

        # self.lm_optimizer = torch.optim.ASGD(self.lm)
        if params.use_multi_process:
            self.otf_start_multiprocessing()

    def otf_start_multiprocessing(self):
        # print("Starting subprocesses for OTF generation ...")

        # initialize subprocesses
        for rank in range(self.num_replicas):
            self.call_async(rank, '_async_otf_init', params=self.params, gpu_id=rank+1)

    def _async_otf_init(self, rank, device_id, params, gpu_id):
        # build model on subprocess

        from copy import deepcopy
        params = deepcopy(params)
        self.params = params
        self.data = None  # do not load data in the CPU threads
        self.iterators = {}
        self.model = Transformer(params).to(torch.device('cuda:' + str(gpu_id)))
        self.type_dict = {'comp': 0, 'simp': 1}

    def otf_sync_params(self):
        # logger.info("Syncing encoder and decoder params for OTF generation ...")

        def get_flat_params(module):
            return torch._utils._flatten_dense_tensors(
                [p.data for p in module.parameters()]
            )

        model_params = get_flat_params(self.model)

        for rank in range(self.num_replicas):
            self.call_async(rank, '_async_otf_sync_params', model_params=model_params)

    def _async_otf_sync_params(self, rank, device_id, model_params):

        def set_flat_params(module, flat):
            params = [p.data for p in module.parameters()]
            for p, f in zip(params, torch._utils._unflatten_dense_tensors(flat, params)):
                p.copy_(f)

        # copy parameters back into modules
        set_flat_params(self.model, model_params)

    def get_iterator(self, iter_name, src_type, tgt_type):
        key = ",".join([x for x in [iter_name, src_type, tgt_type] if x is not None])

        if iter_name == 'encdec' or iter_name == 'otf':
            if tgt_type is None:
                data_loader = self.data['mono'][iter_name][src_type + '_train']
            else:
                data_loader = self.data['para']['train']
        else:
            data_loader = self.data['lm'][src_type + '_train']

        data_iter = data_loader.__iter__()
        self.iterators[key] = data_iter
        return data_iter

    def get_batch(self, iter_name, src_type, tgt_type):
        key = ",".join([x for x in [iter_name, src_type, tgt_type] if x is not None])
        iterator = self.iterators.get(key, None)
        if iterator is None:
            iterator = self.get_iterator(iter_name, src_type, tgt_type)
        try:
            batch = iterator.next()
        except StopIteration:
            iterator = self.get_iterator(iter_name, src_type, tgt_type)
            batch = iterator.next()

        return batch

    def lm_step(self, type):
        if self.lm is None:
            return

        self.lm.train()
        batch = self.get_batch('otf', type, None)
        input_seq, input_pos = map(lambda x: x.to(Constants.device), batch)
        tgt_seq = input_seq[:, 1:].contiguous()
        input_seq = input_seq[:, :-1].contiguous()

        loss = self.lm(input_seq, tgt_seq)
        self.lm_optimizer.zero_grad()
        loss.backward()
        self.lm_optimizer.step()

    def enc_dec_step(self, src_type, tgt_type, use_pointer=False, xe=1, back=False):
        if self.model is None:
            return

        self.model.train()
        if src_type == tgt_type:
            batch = self.get_batch('encdec', src_type, None)
        else:
            batch = self.get_batch('encdec', src_type, tgt_type)

        src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(Constants.device), batch)

        batch_size = src_seq.size(0)
        pred = self.model(src_seq, src_pos, tgt_seq, tgt_pos, self.type_dict[src_type], self.type_dict[tgt_type])
        gold = tgt_seq[:, 1:].contiguous().view(-1)

        if use_pointer:
            loss = F.nll_loss(pred, gold, ignore_index=Constants.PAD, reduction='sum') / batch_size * xe
        else:
            loss = F.cross_entropy(pred, gold, ignore_index=Constants.PAD, reduction='sum') / batch_size * xe

        if src_type == tgt_type:
            self.stat['auto_encoder_'+ src_type + '_loss'].append(loss.item())
        else:
            self.stat[src_type + '_' + tgt_type + '_loss'].append(loss.item())

        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()

        if back:
            pred = self.model(tgt_seq, tgt_pos, src_seq, src_pos, self.type_dict[tgt_type], self.type_dict[src_type])
            gold = src_seq[:, 1:].contiguous().view(-1)
            if use_pointer:
                loss = F.nll_loss(pred, gold, ignore_index=Constants.PAD, reduction='sum') / batch_size * xe
            else:
                loss = F.cross_entropy(pred, gold, ignore_index=Constants.PAD, reduction='sum') / batch_size * xe

            self.model_optimizer.zero_grad()
            loss.backward()
            self.model_optimizer.step()

    def get_reward(self, input_seq, policy_gen, tgt_seq, type='simp'):

        def normal_fkgl(score, type):
            if type == 'simp':
                min_score, max_score = 3, 15
            else:
                min_score, max_score = 15, 30

            if score <= min_score:
                return 0
            elif score >= max_score:
                return 1
            else:
                return (score - min_score) / (max_score - min_score)

        fkgl_rewards = []
        embeddings = []
        if type == 'simp':
            lm_rewards = self.lm.get_ppl_reward(policy_gen)
        else:
            lm_rewards = 0

        for i in range(input_seq.size(0)):
            stop_signal = [Constants.PAD, Constants.EOS, Constants.CBOS, Constants.SBOS]
            
            input_eos = (input_seq[i] == Constants.EOS).nonzero()
            policy_eos = (policy_gen[i] == Constants.EOS).nonzero()

            temp_input_seq = input_seq[i][:input_eos[0][0]+1] if len(input_eos) > 0 and type == 'comp' else input_seq[i]
            temp_policy_gen = policy_gen[i][:policy_eos[0][0]+1] if len(policy_eos) > 0 and type == 'comp' else policy_gen[i]

            next_input_idx = [int(index) for index in temp_input_seq if int(index) not in stop_signal]
            next_policy_idx = [int(index) for index in temp_policy_gen if int(index) not in stop_signal]

            next_input_seq = [self.index2word[index] for index in next_input_idx]
            next_policy_gen = [self.index2word[index] for index in next_policy_idx]

            input_sent, input_align = merge_subword(next_input_seq)
            policy_sent, policy_align = merge_subword(next_policy_gen)

            fkgl = fkgl_score(policy_sent, self.cmu_dict)
            fkgl = normal_fkgl(fkgl, type)
            fkgl_rewards.append(fkgl)

            emb_input = self.sts_model.get_embedding(next_input_idx, input_sent, input_align)
            emb_policy = self.sts_model.get_embedding(next_policy_idx, policy_sent, policy_align)

            embeddings.append(emb_input)
            embeddings.append(emb_policy)

        fkgl_rewards = np.array(fkgl_rewards)
        if type == 'simp':
            fkgl_rewards = 1 - fkgl_rewards 
            
        sts_rewards = self.sts_model.get_reward(np.array(embeddings))

        if type == 'simp':
            rewards = 3 / (1 / (fkgl_rewards + 1e-10) + 1 / (sts_rewards + 1e-10) + 1 / (lm_rewards + 1e-10))
        else:
            rewards = 2 / (1 / (fkgl_rewards + 1e-10) + 1 / (sts_rewards + 1e-10))

        return rewards

    def policy_gradient_step(self, src_type, tgt_type):
        if self.model is None:
            return

        self.model.train()
        batch = self.get_batch('otf', src_type, None)
        input_seq, input_pos = map(lambda x: x.to(Constants.device), batch)
        batch_size = input_seq.size(0)

        policy_gen, log_probs, entropy = self.model.policy_generate(
            src_seq=input_seq,
            src_pos=input_pos,
            src_id=self.type_dict[src_type],
            tgt_id=self.type_dict[tgt_type],
            max_len=self.params.len_max_seq,
            device=Constants.device
        )

        gradient_mask = (policy_gen != Constants.PAD).float()
        rewards = self.get_reward(input_seq, policy_gen, None)

        rewards = torch.Tensor(rewards).float().unsqueeze(-1).to(Constants.device)
        log_probs = torch.cat(log_probs, dim=-1)
        policy_loss = - (log_probs * gradient_mask * rewards).sum() / batch_size
        loss = policy_loss - 0.000 * entropy

        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()

        return (gradient_mask * rewards).sum() / batch_size

    def otf_bt(self, batch, lambda_xe, use_pointer=False, gamma=0):
        """
        On the fly back-translation.
        """
        params = self.params
        src_type, tgt_type, data = batch['src_type'], batch['tgt_type'], batch['data']
        src_seq, tgt_seq, src_pos, tgt_pos = map(lambda x:x.to(Constants.device), data)
        batch_size = src_seq.size(0)
        src_id, tgt_id = self.type_dict[src_type], self.type_dict[tgt_type]

        self.model.train()

        pred = self.model(src_seq, src_pos, tgt_seq, tgt_pos, src_id, tgt_id)
        gold = tgt_seq[:, 1:].contiguous().view(-1)
        if use_pointer:
            loss = F.nll_loss(pred, gold, ignore_index=Constants.PAD, reduction='sum')
        else:
            loss = F.cross_entropy(pred, gold, ignore_index=Constants.PAD, reduction='sum')

        if self.params.rl_finetune:
            prob = F.softmax(pred.view(batch_size, -1, pred.size(-1)), dim=-1)
            mask = (tgt_seq[:, 1:] != Constants.PAD).long()

            _, gred_sent = prob.max(dim=-1)
            gred_sent = gred_sent * mask 
            distribution = Categorical(prob)
            samp_sent = distribution.sample()
            samp_sent = samp_sent * mask 
            log_probs = distribution.log_prob(samp_sent)

            baseline = self.get_reward(input_seq=src_seq, policy_gen=gred_sent, tgt_seq=tgt_seq, type=tgt_type)
            rewards = self.get_reward(input_seq=src_seq, policy_gen=samp_sent, tgt_seq=tgt_seq, type=tgt_type)

            if tgt_type == 'simp':
                avg_reward = rewards.sum() / rewards.shape[0]
                self.rewards_simp.append(float(avg_reward))

            baseline = torch.Tensor(baseline).float().unsqueeze(-1).to(Constants.device)
            rewards = torch.Tensor(rewards).float().unsqueeze(-1).to(Constants.device)

            policy_loss = - (log_probs * (rewards - baseline)).sum() / batch_size

            loss = (1 - gamma) * loss + gamma * policy_loss

        else:
            loss = (1 - gamma) * loss

        self.stat[src_type + '_' + tgt_type + '_otf_loss'].append(loss.item())

        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()

    def otf_bt_gen_async(self, init_cache_size=None):
        # print("Populating initial OTF generation cache ...")
        if init_cache_size is None:
            init_cache_size = self.num_replicas
        cache = [
            self.call_async(rank=i % self.num_replicas, action='_async_otf_bt_gen',
                            result_type='otf_gen', fetch_all=True,
                            batches=self.get_otf_batches())
            for i in range(init_cache_size)
        ]
        while True:
            results = cache[0].gen()
            for rank, _ in results:
                cache.pop(0)  # keep the cache a fixed size
                cache.append(
                    self.call_async(rank=rank, action='_async_otf_bt_gen',
                                    result_type='otf_gen', fetch_all=True,
                                    batches=self.get_otf_batches())
                )
            for _, result in results:
                yield result

    def get_otf_batches(self):
        """
        Create batches for CPU threads.
        """
        simp = self.get_batch('otf', 'simp', None)
        comp = self.get_batch('otf', 'comp', None)
        batches = {'simp': simp, 'comp': comp}

        return batches

    def _async_otf_bt_gen(self, rank, device_id, batches):
        """
        On the fly back-translation (generation step).
        """
        params = self.params
        self.model.eval()

        results = []
        with torch.no_grad():
            simp_seq, simp_pos = map(lambda x: x.to(torch.device('cuda:' + str(rank+1))), batches['simp'])
            comp_seq, comp_pos = map(lambda x: x.to(torch.device('cuda:' + str(rank+1))), batches['comp'])

            # simp -> comp:
            gen_comp, gen_comp_pos = self.model.generate(
                src_seq=simp_seq,
                src_pos=simp_pos,
                src_id=self.type_dict['simp'],
                tgt_id=self.type_dict['comp'],
                max_len=params.len_max_seq,
                mode='otf',
                device=torch.device('cuda:' + str(rank+1))
            )
            # comp -> simp:
            gen_simp, gen_simp_pos = self.model.generate(
                src_seq=comp_seq,
                src_pos=comp_pos,
                src_id=self.type_dict['comp'],
                tgt_id=self.type_dict['simp'],
                max_len=params.len_max_seq,
                mode='otf',
                device=torch.device('cuda:' + str(rank+1))
            )
        
        comp_simp_batch = [gen_comp.cpu(), simp_seq.cpu(), gen_comp_pos.cpu(), simp_pos.cpu()]
        simp_comp_batch = [gen_simp.cpu(), comp_seq.cpu(), gen_simp_pos.cpu(), comp_pos.cpu()]

        results.append(dict([
            ('src_type', 'comp'), ('tgt_type', 'simp'), ('data', comp_simp_batch)
        ]))

        results.append(dict([
            ('src_type', 'simp'), ('tgt_type', 'comp'), ('data', simp_comp_batch)
        ]))

        return (rank, results)

    def iter(self):
        self.n_iter += 1
        self.n_sentences += 4 * self.params.batch_size

        if self.n_iter % 500 == 0 and self.params.rl_finetune:
            write_reward_simp = sum(self.rewards_simp) / len(self.rewards_simp)
            self.rewards_simp.clear()

            path = os.path.join(self.params.dump_path, self.params.name + 'new_reward.log')
            with codecs.open(path, mode='a') as f:
                f.write(str(write_reward_simp) + '\n')

        self.print_stats()

    def print_stats(self, pretrain=False):

        if self.n_iter % 500 == 0 or pretrain:
            simp_loss = 0
            comp_loss = 0
            for key in self.stat.keys():
                key_loss = 0 if len(self.stat[key]) == 0 else sum(self.stat[key]) / len(self.stat[key])
                if key == 'auto_encoder_simp_loss':
                    simp_loss = key_loss
                if key == 'auto_encoder_comp_loss':
                    comp_loss = key_loss
                logger.info(key + ": " + str(key_loss))
                self.stat[key].clear()

            logger.info("generation time: " +  str(self.gen_time))
            logger.info("total time cost: " + str(time.time() - self.start_time))

            self.gen_time = 0
            self.start_time = 0
            return simp_loss, comp_loss

    def end_epoch(self, scores):
        if self.stopping_criterion is not None:
            assert self.stopping_criterion in scores
            if scores[self.stopping_criterion] > self.best_stopping_criterion:
                self.best_stopping_criterion = scores[self.stopping_criterion]
                logger.info("New best validation score: %f" % self.best_stopping_criterion)
                self.decrease_counts = 0
                if scores['bleu'] < 0.20:
                    logger.info("bleu is not enough")
                    self.save_model(self.params.name + 'temp')
                else:
                    logger.info("bleu is enough")
                    self.save_model(self.params.name)
            if scores[self.stopping_criterion] < self.best_stopping_criterion:
                logger.info("Not a better validation score (%i / %i)."
                            % (self.decrease_counts, self.decrease_counts_max))
                self.decrease_counts += 1
                if scores['bleu'] >= 0.20 and scores[self.stopping_criterion] > self.relative_best:
                    logger.info("bleu is enough")
                    self.save_model(self.params.name)
                    self.relative_best = scores[self.stopping_criterion]
            if self.decrease_counts > self.decrease_counts_max:
                logger.info("Stopping criterion has been below its best value more "
                            "than %i epochs. Ending the experiment..." % self.decrease_counts_max)
                return 1
        self.epoch += 1
        self.save_checkpoint()

        return 0

    def save_checkpoint(self):
        checkpoint_data = {
            'model': self.model,
            'optimizer': self.model_optimizer,
            'epoch': self.epoch,
        }
        checkpoint_path = os.path.join(self.params.checkpoint_path, self.params.name)
        logger.info("Saving checkpoint to %s ..." % checkpoint_path)
        torch.save(checkpoint_data, checkpoint_path)

    def reload_checkpoint(self):
        """
        Reload a checkpoint if we find one.
        """
        # reload checkpoint
        checkpoint_path = os.path.join(self.params.checkpoint_path, self.params.name)
        if not os.path.isfile(checkpoint_path):
            return
        logger.warning('Reloading checkpoint from %s ...' % checkpoint_path)
        checkpoint_data = torch.load(checkpoint_path)

        self.model = checkpoint_data['model']
        self.epoch = checkpoint_data['epoch']
        self.model_optimizer = checkpoint_data['optimizer']
        logger.warning('Checkpoint reloaded. Resuming at epoch %i ...' % self.epoch)

    def save_model(self, name):
        path = os.path.join(self.params.dump_path, '%s.pth' % name)
        logger.info('Saving model to %s ...' % path)
        torch.save({
            'model': self.model
        }, path)

    def start_back_translation(self):
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params.lr, betas=(0.5, 0.999))
