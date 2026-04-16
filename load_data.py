from pathlib import Path
import random

import numpy as np


DATA_DIR = Path(__file__).resolve().parent / 'data'


def _data_path(*parts):
    return str(DATA_DIR.joinpath(*parts))


def gen_data(dataset, random_seed):
    train_mat = None
    user_n_item_thresh = 0
    item_n_user_thresh = 0
    valid_is_thresh = 0
    batch_size = 1024
    data_epoch = 10
    learning_rate = 0.001
    if dataset == 'grocery':
        train_mat = np.load(_data_path('grocery.npy'))
        interact_data = {}
        for (u,i,t) in train_mat:
            if not u in interact_data:
                interact_data[u] = []
            if not i in interact_data[u]:
                interact_data[u].append(i)
        user_n_item_thresh = 5
        item_n_user_thresh = 5
        valid_is_thresh = 2
        batch_size = 256
        data_epoch = 25
    if dataset == 'cd':
        train_mat = np.load(_data_path('cd.npy'))
        interact_data = {}
        for (u,i,t) in train_mat:
            if not u in interact_data:
                interact_data[u] = []
            if not i in interact_data[u]:
                interact_data[u].append(i)
        user_n_item_thresh = 5
        item_n_user_thresh = 5
        valid_is_thresh = 2
        batch_size = 256
        data_epoch = 25

    if dataset == 'ml100k':
        lines = [line.strip() for line in open(_data_path('ml100k_u.data'), encoding='iso-8859-1')]
        interact_data = {}
        for line in lines:
            user, item, rating, ts = line.split('\t')
            if not user in interact_data:
                interact_data[user] = []
            interact_data[user].append(item)
        user_n_item_thresh = 5
        item_n_user_thresh = 5
        valid_is_thresh = 2
        batch_size = 2048
        data_epoch = 25

    if dataset == 'ml20m':
        lines = [line.strip() for line in open(_data_path('ml20m_ratings.csv'), encoding='iso-8859-1')]
        interact_data = {}
        for line in lines:
            user, item, rating, ts = line.split(',')
            if not user in interact_data:
                interact_data[user] = []
            interact_data[user].append(item)
        user_n_item_thresh = 500
        item_n_user_thresh = 500
        valid_is_thresh = 500
        batch_size = 10000
        data_epoch = 10

    if dataset == 'adressa':
        interact_data = np.load(_data_path('gnud_user_rt_news.npy'), allow_pickle=True).item()
        user_n_item_thresh = 5
        item_n_user_thresh = 5
        valid_is_thresh = 2
        batch_size = 2048
        data_epoch = 10
        learning_rate = 0.0001

    if dataset == 'gowalla':
        lines = [line.strip() for line in open(_data_path('loc-gowalla_totalCheckins.txt'))]
        interact_data = {}
        for line in lines:
            user, time, long, lat, item = line.split('\t')
            if not user in interact_data:
                interact_data[user] = []
            interact_data[user].append(item)
        user_n_item_thresh = 100
        item_n_user_thresh = 100
        valid_is_thresh = 100
        batch_size = 2048
        data_epoch = 10

    temp = {}
    for u in interact_data:
        if len(interact_data[u]) > user_n_item_thresh and len(interact_data[u]) < user_n_item_thresh + 1000:
            temp[u] = interact_data[u]
    interact_data = temp

    item_interaction_count = {}
    for u in interact_data:
        for i in interact_data[u]:
            if not i in item_interaction_count:
                item_interaction_count[i] = 0
            item_interaction_count[i] += 1

    valid_item = set()
    for i in item_interaction_count:
        if item_interaction_count[i] > item_n_user_thresh:
            valid_item.add(i)
    print('number of valid items:', len(valid_item))

    temp = {}
    for u in interact_data:
        valid_is = [i for i in interact_data[u] if i in valid_item]
        if len(valid_is) > valid_is_thresh:
            temp[u] = valid_is
    interact_data = temp
    print('number of users with valid interactions:', len(interact_data))

    users = interact_data.keys()
    items = set()
    interact_count = 0
    all_items = []
    for u in users:
        for i in interact_data[u]:
            if not i in valid_item:
                continue
            items.add(i)
            all_items.append(i)
            interact_count += 1
    print('users', len(users))
    print('items', len(items))
    print('interactions', interact_count)

    num_users = len(users)
    num_items = len(items)
    num_nodes = num_users + num_items
    emb_dim = 20

    user_to_id = {}
    for idx,u in enumerate(sorted(list(users))):
        user_to_id[u] = idx
    item_to_id = {}
    for idx,i in enumerate(sorted(list(items))):
        item_to_id[i] = idx

    a_mat = np.zeros((num_users + num_items, num_users + num_items))
    for u in users:
        u_idx = user_to_id[u]
        for i in interact_data[u]:
            i_idx = item_to_id[i]
            a_mat[u_idx, num_users+i_idx] = 1
            a_mat[num_users+i_idx, u_idx] = 1


    item_pop = {}
    for item,row in enumerate(a_mat[num_users:, :num_users]):
        pop = sum(row)
        item_pop[item] = pop

    # Leave-One-Out Test Set Creation
    r_gen = random.Random(random_seed)  # For reproducibility
    test_mask = np.zeros_like(a_mat, dtype=bool)
    train_mask = np.ones_like(a_mat, dtype=bool)
    test_mask_div = np.zeros_like(a_mat, dtype=bool)

    def normalize(values):
        total = sum(values)
        return [x / total for x in values]

    for user_id in range(num_users):
        # Find indices of all items this user has interacted with
        interactions = np.where(a_mat[user_id, num_users:] == 1)[0]
        interactions += num_users  # Adjust index to full matrix

        # Select one random interaction for testing
        test_item = r_gen.choice(interactions)
        test_mask[user_id, test_item] = True
        test_mask[test_item, user_id] = True  # Symmetric for undirected graph

        # Remove this interaction from training data
        train_mask[user_id, test_item] = False
        train_mask[test_item, user_id] = False

        # create diversify test
        inv_pops = [1/item_pop[i-num_users] for i in interactions]
        prob = normalize(inv_pops)
        test_item_div = r_gen.choices(interactions, weights=prob, k=1)[0]
        test_mask_div[user_id, test_item_div] = True
        test_mask_div[test_item_div, user_id] = True  # Symmetric for undirected graph
        train_mask[user_id, test_item_div] = False
        train_mask[test_item_div, user_id] = False

        
    def mask_adjacency_matrix(adj_matrix, mask):
        """Apply a mask to the adjacency matrix to zero out disallowed connections."""
        masked_adj_matrix = adj_matrix.copy()
        masked_adj_matrix[~mask] = 0
        return masked_adj_matrix

    # Create masked adjacency matrices for training and testing
    train_a_mat = mask_adjacency_matrix(a_mat, train_mask)
    test_a_mat = mask_adjacency_matrix(a_mat, test_mask)
    test_a_mat_div = mask_adjacency_matrix(a_mat, test_mask_div)


    train_interact_data = {}
    train_interact_num = 0
    for u,row in enumerate(train_a_mat):
        if u == num_users:
            break
        items = np.where(train_a_mat[u, num_users:] > 0)[0]
        train_interact_data[u] = items
        train_interact_num += len(items)
    print("number of train interactions in train set:", train_interact_num)

    data = {'num_users': num_users, 'num_items': num_items, 'train_a1' : train_a_mat, 'test_a1' : test_a_mat, 'test_a1_div' : test_a_mat_div}

    return data, batch_size, data_epoch, learning_rate



