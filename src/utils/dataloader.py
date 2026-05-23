# coding: utf-8
# @email: enoche.chow@gmail.com
"""
Wrap dataset into dataloader
################################################
"""
import math
import torch
import random
import numpy as np
from logging import getLogger
from scipy.sparse import coo_matrix


class AbstractDataLoader(object):
    """:class:`AbstractDataLoader` is an abstract object which would return a batch of data which is loaded by
    :class:`~recbole.data.interaction.Interaction` when it is iterated.
    And it is also the ancestor of all other dataloader.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        batch_size (int, optional): The batch_size of dataloader. Defaults to ``1``.
        dl_format (InputType, optional): The input type of dataloader. Defaults to
            :obj:`~recbole.utils.enum_type.InputType.POINTWISE`.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.

    Attributes:
        dataset (Dataset): The dataset of this dataloader.
        shuffle (bool): If ``True``, dataloader will shuffle before every epoch.
        real_time (bool): If ``True``, dataloader will do data pre-processing,
            such as neg-sampling and data-augmentation.
        pr (int): Pointer of dataloader.
        step (int): The increment of :attr:`pr` for each batch.
        batch_size (int): The max interaction number for all batch.
    """
    def __init__(self, config, dataset, additional_dataset=None,
                 batch_size=1, neg_sampling=False, shuffle=False):
        self.config = config
        self.logger = getLogger()
        self.dataset = dataset
        self.dataset_bk = self.dataset.copy(self.dataset.df)
        # if config['model_type'] == ModelType.GENERAL:
        #     self.dataset.df.drop(self.dataset.ts_id, inplace=True, axis=1)
        # elif config['model_type'] == ModelType.SEQUENTIAL:
        #     # sort instances
        #     pass
        self.additional_dataset = additional_dataset
        self.batch_size = batch_size
        self.step = batch_size
        self.shuffle = shuffle
        self.neg_sampling = neg_sampling
        self.device = config['device']

        self.sparsity = 1 - self.dataset.inter_num / self.dataset.user_num / self.dataset.item_num
        self.pr = 0
        self.inter_pr = 0
        self._refresh_cached_arrays()

    def _refresh_cached_arrays(self):
        self.uid_array = self.dataset.df[self.dataset.uid_field].to_numpy(dtype=np.int64, copy=True)
        self.iid_array = self.dataset.df[self.dataset.iid_field].to_numpy(dtype=np.int64, copy=True)

    def pretrain_setup(self):
        """This function can be used to deal with some problems after essential args are initialized,
        such as the batch-size-adaptation when neg-sampling is needed, and so on. By default, it will do nothing.
        """
        pass

    def data_preprocess(self):
        """This function is used to do some data preprocess, such as pre-neg-sampling and pre-data-augmentation.
        By default, it will do nothing.
        """
        pass

    def __len__(self):
        return math.ceil(self.pr_end / self.step)

    def __iter__(self):
        if self.shuffle:
            self._shuffle()
        return self

    def __next__(self):
        if self.pr >= self.pr_end:
            self.pr = 0
            self.inter_pr = 0
            raise StopIteration()
        return self._next_batch_data()

    @property
    def pr_end(self):
        """This property marks the end of dataloader.pr which is used in :meth:`__next__()`."""
        raise NotImplementedError('Method [pr_end] should be implemented')

    def _shuffle(self):
        """Shuffle the order of data, and it will be called by :meth:`__iter__()` if self.shuffle is True.
        """
        raise NotImplementedError('Method [shuffle] should be implemented.')

    def _next_batch_data(self):
        """Assemble next batch of data in form of Interaction, and return these data.

        Returns:
            Interaction: The next batch of data.
        """
        raise NotImplementedError('Method [next_batch_data] should be implemented.')


class TrainDataLoader(AbstractDataLoader):
    """
    General dataloader with negative sampling.
    """
    def __init__(self, config, dataset, batch_size=1, shuffle=False):
        super().__init__(config, dataset, additional_dataset=None,
                         batch_size=batch_size, neg_sampling=True, shuffle=shuffle)

        # special for training dataloader
        self.history_items_per_u = dict()
        # full items in training.
        self.all_items = self.dataset.df[self.dataset.iid_field].unique().tolist()
        self.all_uids = self.dataset.df[self.dataset.uid_field].unique()
        self.all_users = self.all_uids.tolist()
        self.all_items_set = set(self.all_items)
        self.all_users_set = set(self.all_uids)
        self.all_item_len = len(self.all_items)
        # if full sampling
        self.use_full_sampling = config['use_full_sampling']

        if config['use_neg_sampling']:
            if self.use_full_sampling:
                self.sample_func = self._get_full_uids_sample
            else:
                self.sample_func = self._get_neg_sample
        else:
            self.sample_func = self._get_non_neg_sample

        self._get_history_items_u()
        self.neighborhood_loss_required = config['use_neighborhood_loss']
        if self.neighborhood_loss_required:
            self.history_users_per_i = {}
            self._get_history_users_i()
            self.user_user_dict = self._get_my_neighbors(self.config['USER_ID_FIELD'])
            self.item_item_dict = self._get_my_neighbors(self.config['ITEM_ID_FIELD'])

    def pretrain_setup(self):
        """
        Reset dataloader. Outputing the same positive & negative samples with each training.
        :return:
        """
        # sort & random
        if self.shuffle:
            self.dataset = self.dataset_bk.copy(self.dataset_bk.df)
            self._refresh_cached_arrays()
        self.all_items.sort()
        if self.use_full_sampling:
            self.all_uids.sort()
        self.all_users = self.all_uids.tolist()
        random.shuffle(self.all_items)
        # reorder dataset as default (chronological order)
        #self.dataset.sort_by_chronological()

    def inter_matrix(self, form='coo', value_field=None):
        """Get sparse matrix that describe interactions between user_id and item_id.

        Sparse matrix has shape (user_num, item_num).

        For a row of <src, tgt>, ``matrix[src, tgt] = 1`` if ``value_field`` is ``None``,
        else ``matrix[src, tgt] = self.inter_feat[src, tgt]``.

        Args:
            form (str, optional): Sparse matrix format. Defaults to ``coo``.
            value_field (str, optional): Data of sparse matrix, which should exist in ``df_feat``.
                Defaults to ``None``.

        Returns:
            scipy.sparse: Sparse matrix in form ``coo`` or ``csr``.
        """
        if not self.dataset.uid_field or not self.dataset.iid_field:
            raise ValueError('dataset doesn\'t exist uid/iid, thus can not converted to sparse matrix')
        return self._create_sparse_matrix(self.dataset.df, self.dataset.uid_field,
                                          self.dataset.iid_field, form, value_field)

    def _create_sparse_matrix(self, df_feat, source_field, target_field, form='coo', value_field=None):
        """Get sparse matrix that describe relations between two fields.

        Source and target should be token-like fields.

        Sparse matrix has shape (``self.num(source_field)``, ``self.num(target_field)``).

        For a row of <src, tgt>, ``matrix[src, tgt] = 1`` if ``value_field`` is ``None``,
        else ``matrix[src, tgt] = df_feat[value_field][src, tgt]``.

        Args:
            df_feat (pandas.DataFrame): Feature where src and tgt exist.
            form (str, optional): Sparse matrix format. Defaults to ``coo``.
            value_field (str, optional): Data of sparse matrix, which should exist in ``df_feat``.
                Defaults to ``None``.

        Returns:
            scipy.sparse: Sparse matrix in form ``coo`` or ``csr``.
        """
        src = df_feat[source_field].values
        tgt = df_feat[target_field].values
        if value_field is None:
            data = np.ones(len(df_feat))
        else:
            if value_field not in df_feat.columns:
                raise ValueError('value_field [{}] should be one of `df_feat`\'s features.'.format(value_field))
            data = df_feat[value_field].values
        mat = coo_matrix((data, (src, tgt)), shape=(self.dataset.user_num, self.dataset.item_num))

        if form == 'coo':
            return mat
        elif form == 'csr':
            return mat.tocsr()
        else:
            raise NotImplementedError('sparse matrix format [{}] has not been implemented.'.format(form))

    @property
    def pr_end(self):
        if self.use_full_sampling:
            return len(self.all_uids)
        return len(self.dataset)

    def _shuffle(self):
        self.dataset.shuffle()
        self._refresh_cached_arrays()
        if self.use_full_sampling:
            np.random.shuffle(self.all_uids)
            self.all_users = self.all_uids.tolist()

    def _next_batch_data(self):
        return self.sample_func()

    def _get_neg_sample(self):
        start = self.pr
        end = min(self.pr + self.step, self.pr_end)
        user_ids = self.uid_array[start:end]
        item_ids = self.iid_array[start:end]
        self.pr = end
        # to tensor
        user_tensor = torch.as_tensor(user_ids, dtype=torch.long, device=self.device)
        item_tensor = torch.as_tensor(item_ids, dtype=torch.long, device=self.device)
        batch_tensor = torch.cat((torch.unsqueeze(user_tensor, 0),
                                  torch.unsqueeze(item_tensor, 0)))
        # sampling negative items only in the dataset (train)
        neg_ids = self._sample_neg_ids(user_ids)
        # for neighborhood loss
        if self.neighborhood_loss_required:
            pos_neighbors, neg_neighbors = self._get_neighborhood_samples(item_ids, self.config['ITEM_ID_FIELD'])

            batch_tensor = torch.cat((batch_tensor, neg_ids.unsqueeze(0),
                                      pos_neighbors.unsqueeze(0), neg_neighbors.unsqueeze(0)))

        # merge negative samples
        else:
            batch_tensor = torch.cat((batch_tensor, neg_ids.unsqueeze(0)))

        return batch_tensor

    def _get_non_neg_sample(self):
        start = self.pr
        end = min(self.pr + self.step, self.pr_end)
        user_ids = self.uid_array[start:end]
        item_ids = self.iid_array[start:end]
        self.pr = end
        # to tensor
        user_tensor = torch.as_tensor(user_ids, dtype=torch.long, device=self.device)
        item_tensor = torch.as_tensor(item_ids, dtype=torch.long, device=self.device)
        batch_tensor = torch.cat((torch.unsqueeze(user_tensor, 0),
                                  torch.unsqueeze(item_tensor, 0)))
        return batch_tensor

    def _get_full_uids_sample(self):
        end = min(self.pr + self.step, self.pr_end)
        user_tensor = torch.as_tensor(self.all_uids[self.pr:end], dtype=torch.long, device=self.device)
        self.pr = end
        return user_tensor

    def _sample_neg_ids(self, u_ids):
        neg_ids = []
        for u in u_ids:
            # random 1 item
            iid = self._random()
            while iid in self.history_items_per_u[u]:
                iid = self._random()
            neg_ids.append(iid)
        return torch.as_tensor(neg_ids, dtype=torch.long, device=self.device)

    def _get_my_neighbors(self, id_str):
        ret_dict = {}
        a2b_dict = self.history_items_per_u if id_str == self.config['USER_ID_FIELD'] else self.history_users_per_i
        b2a_dict = self.history_users_per_i if id_str == self.config['USER_ID_FIELD'] else self.history_items_per_u
        for i, j in a2b_dict.items():
            k = set()
            for m in j:
                k |= b2a_dict.get(m, set()).copy()
            k.discard(i)                        # remove myself
            ret_dict[i] = k
        return ret_dict

    def _get_neighborhood_samples(self, ids, id_str):
        a2a_dict = self.user_user_dict if id_str == self.config['USER_ID_FIELD'] else self.item_item_dict
        all_set = self.all_users_set if id_str == self.config['USER_ID_FIELD'] else self.all_items_set
        all_values = self.all_users if id_str == self.config['USER_ID_FIELD'] else self.all_items
        pos_ids, neg_ids = [], []
        for i in ids:
            pos_ids_my = a2a_dict[i]
            if len(pos_ids_my) <= 0 or len(pos_ids_my)/len(all_set) > 0.8:
                pos_ids.append(0)
                neg_ids.append(0)
                continue
            pos_id = random.choice(tuple(pos_ids_my))
            pos_ids.append(pos_id)
            neg_id = random.choice(all_values)
            while neg_id in pos_ids_my:
                neg_id = random.choice(all_values)
            neg_ids.append(neg_id)
        return (torch.as_tensor(pos_ids, dtype=torch.long, device=self.device),
                torch.as_tensor(neg_ids, dtype=torch.long, device=self.device))

    def _random(self):
        rd_id = random.sample(self.all_items, 1)[0]
        return rd_id

    def _get_history_items_u(self):
        uid_field = self.dataset.uid_field
        iid_field = self.dataset.iid_field
        # load avail items for all uid
        uid_freq = self.dataset.df.groupby(uid_field)[iid_field]
        for u, u_ls in uid_freq:
            self.history_items_per_u[u] = set(u_ls.values)
        return self.history_items_per_u

    def _get_history_users_i(self):
        uid_field = self.dataset.uid_field
        iid_field = self.dataset.iid_field
        # load avail items for all uid
        iid_freq = self.dataset.df.groupby(iid_field)[uid_field]
        for i, u_ls in iid_freq:
            self.history_users_per_i[i] = set(u_ls.values)
        return self.history_users_per_i


class EvalDataLoader(AbstractDataLoader):
    """
        additional_dataset: training dataset in evaluation
    """
    def __init__(self, config, dataset, additional_dataset=None,
                 batch_size=1, shuffle=False):
        super().__init__(config, dataset, additional_dataset=additional_dataset,
                         batch_size=batch_size, neg_sampling=False, shuffle=shuffle)

        if additional_dataset is None:
            raise ValueError('Training datasets is nan')
        self.eval_items_per_u = []
        self.eval_len_list = []
        self.train_pos_len_list = []

        self.eval_u = self.dataset.df[self.dataset.uid_field].unique()
        self.eval_user_np = np.asarray(self.eval_u, dtype=np.int64)
        self._build_extended_eval_stats()
        # special for eval dataloader
        self.pos_items_per_u = self._get_pos_items_per_u(self.eval_u)
        self._get_eval_items_per_u(self.eval_u)
        self.train_pos_len_list = np.asarray(self.train_pos_len_list, dtype=np.int64)
        self.train_pos_offsets = np.concatenate(([0], np.cumsum(self.train_pos_len_list)))
        # to device
        self.eval_u = torch.as_tensor(self.eval_u, dtype=torch.long, device=self.device)

    @property
    def pr_end(self):
        return self.eval_u.shape[0]

    def _shuffle(self):
        self.dataset.shuffle()

    def _build_extended_eval_stats(self):
        uid_field = self.additional_dataset.uid_field
        iid_field = self.additional_dataset.iid_field
        item_ids = self.additional_dataset.df[iid_field].to_numpy(dtype=np.int64)
        user_ids = self.additional_dataset.df[uid_field].to_numpy(dtype=np.int64)

        self.train_item_num = self.additional_dataset.get_item_num()
        self.item_popularity = np.bincount(item_ids, minlength=self.train_item_num).astype(np.float64)

        order = np.lexsort((np.arange(self.train_item_num), self.item_popularity))
        n_items = len(order)
        tail_ratio = self.config['item_tail_ratio']
        head_ratio = self.config['item_head_ratio']
        tail_ratio = 0.5 if tail_ratio is None else tail_ratio
        head_ratio = 0.2 if head_ratio is None else head_ratio
        tail_count = int(np.ceil(n_items * tail_ratio))
        head_count = int(np.ceil(n_items * head_ratio))
        tail_count = min(max(tail_count, 0), n_items)
        head_count = min(max(head_count, 0), max(n_items - tail_count, 0))

        self.tail_item_mask = np.zeros(self.train_item_num, dtype=bool)
        self.mid_item_mask = np.zeros(self.train_item_num, dtype=bool)
        self.head_item_mask = np.zeros(self.train_item_num, dtype=bool)
        if tail_count > 0:
            self.tail_item_mask[order[:tail_count]] = True
        if head_count > 0:
            self.head_item_mask[order[n_items - head_count:]] = True
        self.mid_item_mask = ~(self.tail_item_mask | self.head_item_mask)

        user_history_len = np.bincount(user_ids, minlength=self.additional_dataset.get_user_num()).astype(np.int64)
        eval_history_len = user_history_len[self.eval_user_np]
        cold_max = self.config['cold_user_max_interactions']
        hot_min = self.config['hot_user_min_interactions']
        cold_max = 5 if cold_max is None else cold_max
        hot_min = 21 if hot_min is None else hot_min
        self.cold_user_mask = eval_history_len <= cold_max
        self.hot_user_mask = eval_history_len >= hot_min
        self.warm_user_mask = ~(self.cold_user_mask | self.hot_user_mask)

    def _next_batch_data(self):
        start = self.pr
        end = min(self.pr + self.step, self.pr_end)
        inter_start = int(self.train_pos_offsets[start])
        inter_end = int(self.train_pos_offsets[end])
        batch_users = self.eval_u[start:end]
        batch_mask_matrix = self.pos_items_per_u[:, inter_start:inter_end].clone()
        # user_ids to index
        batch_mask_matrix[0] -= start
        self.inter_pr = inter_end
        self.pr = end

        return [batch_users, batch_mask_matrix]

    def _get_pos_items_per_u(self, eval_users):
        """
        history items in training dataset.
        masking out positive items in evaluation
        :return:
        user_id - item_ids matrix
        [[0, 0, ... , 1, ...],
         [0, 1, ... , 0, ...]]
        """
        uid_field = self.additional_dataset.uid_field
        iid_field = self.additional_dataset.iid_field
        # load avail items for all uid
        uid_freq = self.additional_dataset.df.groupby(uid_field)[iid_field]
        u_ids = []
        i_ids = []
        for i, u in enumerate(eval_users):
            u_ls = uid_freq.get_group(u).values
            i_len = len(u_ls)
            self.train_pos_len_list.append(i_len)
            u_ids.extend([i]*i_len)
            i_ids.extend(u_ls)
        return torch.as_tensor([u_ids, i_ids], dtype=torch.long, device=self.device)

    def _get_eval_items_per_u(self, eval_users):
        """
        get evaluated items for each u
        :return:
        """
        uid_field = self.dataset.uid_field
        iid_field = self.dataset.iid_field
        # load avail items for all uid
        uid_freq = self.dataset.df.groupby(uid_field)[iid_field]
        for u in eval_users:
            u_ls = uid_freq.get_group(u).values
            self.eval_len_list.append(len(u_ls))
            self.eval_items_per_u.append(u_ls)
        self.eval_len_list = np.asarray(self.eval_len_list)

    # return pos_items for each u
    def get_eval_items(self):
        return self.eval_items_per_u

    def get_eval_len_list(self):
        return self.eval_len_list

    def get_eval_users(self):
        return self.eval_u.cpu()

    def get_train_item_num(self):
        return self.train_item_num

    def get_item_popularity(self):
        return self.item_popularity

    def get_item_bucket_masks(self):
        return {
            'head': self.head_item_mask,
            'mid': self.mid_item_mask,
            'tail': self.tail_item_mask,
        }

    def get_user_group_masks(self):
        return {
            'cold': self.cold_user_mask,
            'warm': self.warm_user_mask,
            'hot': self.hot_user_mask,
        }


