# coding: utf-8
# @email: enoche.chow@gmail.com
"""
################################
"""
import os
import numpy as np
import pandas as pd
import torch
from utils.metrics import metrics_dict
from torch.nn.utils.rnn import pad_sequence
from utils.utils import get_local_time


# These metrics are typical in topk recommendations
topk_metrics = {metric.lower(): metric for metric in ['Recall', 'Recall2', 'Precision', 'NDCG', 'MAP']}


class TopKEvaluator(object):
    r"""TopK Evaluator is mainly used in ranking tasks. Now, we support six topk metrics which
    contain `'Hit', 'Recall', 'MRR', 'Precision', 'NDCG', 'MAP'`.

    Note:
        The metrics used calculate group-based metrics which considers the metrics scores averaged
        across users. Some of them are also limited to k.

    """

    def __init__(self, config):
        self.config = config
        self.metrics = config['metrics']
        self.topk = config['topk']
        self.save_recom_result = config['save_recommended_topk']
        self._check_args()

    def collect(self, interaction, scores_tensor, full=False):
        """collect the topk intermediate result of one batch, this function mainly
        implements padding and TopK finding. It is called at the end of each batch

        Args:
            interaction (Interaction): :class:`AbstractEvaluator` of the batch
            scores_tensor (tensor): the tensor of model output with size of `(N, )`
            full (bool, optional): whether it is full sort. Default: False.

        """
        user_len_list = interaction.user_len_list
        if full is True:
            scores_matrix = scores_tensor.view(len(user_len_list), -1)
        else:
            scores_list = torch.split(scores_tensor, user_len_list, dim=0)
            scores_matrix = pad_sequence(scores_list, batch_first=True, padding_value=-np.inf)  # nusers x items

        # get topk
        _, topk_index = torch.topk(scores_matrix, max(self.topk), dim=-1)  # nusers x k

        return topk_index

    def evaluate(self, batch_matrix_list, eval_data, is_test=False, idx=0):
        """calculate the metrics of all batches. It is called at the end of each epoch

        Args:
            batch_matrix_list (list): the results of all batches
            eval_data (Dataset): the class of test data
            is_test: in testing?

        Returns:
            dict: such as ``{'Hit@20': 0.3824, 'Recall@20': 0.0527, 'Hit@10': 0.3153, 'Recall@10': 0.0329}``

        """
        pos_items = eval_data.get_eval_items()
        pos_len_list = eval_data.get_eval_len_list()
        topk_index = torch.cat(batch_matrix_list, dim=0).cpu().numpy()
        # if save recommendation result?
        if self.save_recom_result and is_test:
            dataset_name = self.config['dataset']
            model_name = self.config['model']
            max_k = max(self.topk)
            dir_name = os.path.abspath(self.config['recommend_topk'])
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            file_path = os.path.join(dir_name, '{}-{}-idx{}-top{}-{}.csv'.format(
                model_name, dataset_name, idx, max_k, get_local_time()))
            x_df = pd.DataFrame(topk_index)
            x_df.insert(0, 'id', eval_data.get_eval_users())
            x_df.columns = ['id']+['top_'+str(i) for i in range(max_k)]
            x_df = x_df.astype(int)
            x_df.to_csv(file_path, sep='\t', index=False)
        assert len(pos_len_list) == len(topk_index)
        # if recom right?
        bool_rec_matrix = []
        for m, n in zip(pos_items, topk_index):
            bool_rec_matrix.append([True if i in m else False for i in n])
        bool_rec_matrix = np.asarray(bool_rec_matrix)

        # get metrics
        metric_dict = {}
        result_list = self._calculate_metrics(pos_len_list, bool_rec_matrix)
        for metric, value in zip(self.metrics, result_list):
            for k in self.topk:
                key = '{}@{}'.format(metric, k)
                metric_dict[key] = round(value[k - 1], 4)
        extended_enabled = self.config['extended_eval_metrics']
        if is_test and (extended_enabled is None or extended_enabled):
            metric_dict.update(self._calculate_extended_metrics(topk_index, bool_rec_matrix, pos_items, pos_len_list, eval_data))
        return metric_dict

    def _calculate_extended_metrics(self, topk_index, bool_rec_matrix, pos_items, pos_len_list, eval_data):
        metric_dict = {}
        item_num = eval_data.get_train_item_num()
        item_popularity = eval_data.get_item_popularity()
        item_bucket_masks = eval_data.get_item_bucket_masks()
        user_group_masks = eval_data.get_user_group_masks()
        pos_sets = [set(np.asarray(items, dtype=np.int64).tolist()) for items in pos_items]

        for k in self.topk:
            topk_at_k = topk_index[:, :k]
            hit_at_k = bool_rec_matrix[:, :k]
            metric_dict[f'hit@{k}'] = round(float(np.any(hit_at_k, axis=1).mean()), 4) if len(hit_at_k) else 0.0
            metric_dict[f'coverage@{k}'] = round(float(len(np.unique(topk_at_k)) / item_num), 4) if item_num > 0 else 0.0
            metric_dict[f'avg_pop@{k}'] = round(float(item_popularity[topk_at_k].mean()), 4) if topk_at_k.size else 0.0
            metric_dict[f'gini@{k}'] = round(self._gini_from_topk(topk_at_k, item_num), 4)
            tail_mask = item_bucket_masks['tail']
            metric_dict[f'tail_ratio@{k}'] = round(float(tail_mask[topk_at_k].mean()), 4) if topk_at_k.size else 0.0
            for bucket_name in ['head', 'mid', 'tail']:
                metric_dict[f'{bucket_name}_recall@{k}'] = round(
                    self._bucket_recall_at_k(topk_at_k, pos_sets, item_bucket_masks[bucket_name]), 4
                )

        ndcg_k = 20
        topk_at_20 = topk_index[:, :ndcg_k]
        for bucket_name in ['head', 'mid', 'tail']:
            metric_dict[f'{bucket_name}_ndcg@20'] = round(
                self._bucket_ndcg_at_k(topk_at_20, pos_sets, item_bucket_masks[bucket_name], ndcg_k), 4
            )

        recall_k = 20
        user_recall_at_20 = bool_rec_matrix[:, :recall_k].sum(axis=1) / pos_len_list
        for group_name in ['cold', 'warm', 'hot']:
            group_mask = user_group_masks[group_name]
            if np.any(group_mask):
                metric_dict[f'{group_name}_recall@20'] = round(float(user_recall_at_20[group_mask].mean()), 4)
            else:
                metric_dict[f'{group_name}_recall@20'] = 0.0
        return metric_dict

    @staticmethod
    def _gini_from_topk(topk_at_k, item_num):
        if item_num <= 0 or topk_at_k.size == 0:
            return 0.0
        counts = np.bincount(topk_at_k.reshape(-1), minlength=item_num).astype(np.float64)
        total = counts.sum()
        if total <= 0:
            return 0.0
        sorted_counts = np.sort(counts)
        index = np.arange(1, item_num + 1, dtype=np.float64)
        return float((2.0 * np.sum(index * sorted_counts)) / (item_num * total) - (item_num + 1.0) / item_num)

    @staticmethod
    def _bucket_recall_at_k(topk_at_k, pos_sets, bucket_mask):
        recall_values = []
        bucket_items = set(np.where(bucket_mask)[0].tolist())
        for rec_items, user_pos in zip(topk_at_k, pos_sets):
            relevant = user_pos & bucket_items
            if not relevant:
                continue
            hits = len(set(rec_items.tolist()) & relevant)
            recall_values.append(hits / len(relevant))
        return float(np.mean(recall_values)) if recall_values else 0.0

    @staticmethod
    def _bucket_ndcg_at_k(topk_at_k, pos_sets, bucket_mask, k):
        ndcg_values = []
        bucket_items = set(np.where(bucket_mask)[0].tolist())
        discounts = 1.0 / np.log2(np.arange(2, k + 2))
        for rec_items, user_pos in zip(topk_at_k, pos_sets):
            relevant = user_pos & bucket_items
            if not relevant:
                continue
            gains = np.asarray([item in relevant for item in rec_items[:k]], dtype=np.float64)
            dcg = float(np.sum(gains * discounts[:len(gains)]))
            ideal_len = min(len(relevant), k)
            idcg = float(np.sum(discounts[:ideal_len]))
            ndcg_values.append(dcg / idcg if idcg > 0 else 0.0)
        return float(np.mean(ndcg_values)) if ndcg_values else 0.0

    def _check_args(self):
        # Check metrics
        if isinstance(self.metrics, (str, list)):
            if isinstance(self.metrics, str):
                self.metrics = [self.metrics]
        else:
            raise TypeError('metrics must be str or list')

        # Convert metric to lowercase
        for m in self.metrics:
            if m.lower() not in topk_metrics:
                raise ValueError("There is no user grouped topk metric named {}!".format(m))
        self.metrics = [metric.lower() for metric in self.metrics]

        # Check topk:
        if isinstance(self.topk, (int, list)):
            if isinstance(self.topk, int):
                self.topk = [self.topk]
            for topk in self.topk:
                if topk <= 0:
                    raise ValueError(
                        'topk must be a positive integer or a list of positive integers, but get `{}`'.format(topk))
        else:
            raise TypeError('The topk must be a integer, list')

    def _calculate_metrics(self, pos_len_list, topk_index):
        """integrate the results of each batch and evaluate the topk metrics by users

        Args:
            pos_len_list (list): a list of users' positive items
            topk_index (np.ndarray): a matrix which contains the index of the topk items for users
        Returns:
            np.ndarray: a matrix which contains the metrics result
        """
        result_list = []
        for metric in self.metrics:
            metric_fuc = metrics_dict[metric.lower()]
            result = metric_fuc(topk_index, pos_len_list)
            result_list.append(result)
        return np.stack(result_list, axis=0)

    def __str__(self):
        mesg = 'The TopK Evaluator Info:\n' + '\tMetrics:[' + ', '.join(
            [topk_metrics[metric.lower()] for metric in self.metrics]) \
               + '], TopK:[' + ', '.join(map(str, self.topk)) + ']'
        return mesg
