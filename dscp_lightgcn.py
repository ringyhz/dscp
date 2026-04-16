import argparse
import random
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import tensorflow as tf

import evaluation
import load_data


DEFAULT_KS = [5, 10, 20, 50, 100]
SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description='Train and evaluate DSCP LightGCN.')
    parser.add_argument('--dataset', default='grocery', help='Dataset name supported by load_data.py.')
    parser.add_argument('--embedding-dim', type=int, default=16, help='Embedding dimension for the final model.')
    parser.add_argument(
        '--random-seeds',
        type=int,
        nargs='+',
        default=[111, 222, 333, 444, 555],
        help='Random seeds used for repeated runs.',
    )
    parser.add_argument('--epochs', type=int, default=None, help='Override the dataset-specific epoch count.')
    parser.add_argument('--batch-size', type=int, default=None, help='Override the dataset-specific batch size.')
    parser.add_argument('--learning-rate', type=float, default=None, help='Override the dataset-specific learning rate.')
    parser.add_argument('--alpha', type=float, default=0.1, help='Weight for the personal preference auxiliary loss.')
    parser.add_argument('--beta', type=float, default=0.1, help='Weight for the public preference auxiliary loss.')
    parser.add_argument('--layers', type=int, default=3, help='Number of LightGCN propagation layers.')
    parser.add_argument(
        '--output',
        default=None,
        help='Optional output .npy file. Defaults to DSCP_LightGCN-<dataset>-results.npy next to the script.',
    )
    return parser.parse_args()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def create_optimizer(learning_rate):
    legacy_optimizers = getattr(tf.keras.optimizers, 'legacy', None)
    if legacy_optimizers is not None and hasattr(legacy_optimizers, 'Adam'):
        return legacy_optimizers.Adam(learning_rate=learning_rate)
    return tf.keras.optimizers.Adam(learning_rate=learning_rate)


def normalize_adj(adj_matrix):
    rowsum = np.array(adj_matrix.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5, where=rowsum != 0)
    d_inv_sqrt[~np.isfinite(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj_matrix.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def to_sparse_tensor(adj_matrix):
    normalized = normalize_adj(sp.csr_matrix(adj_matrix))
    indices = np.column_stack((normalized.row, normalized.col))
    return tf.sparse.SparseTensor(
        indices=indices,
        values=normalized.data.astype(np.float32),
        dense_shape=normalized.shape,
    )


def build_item_popularity(train_a_mat, num_users, num_items):
    item_pop = np.sum(train_a_mat[num_users:, :num_users], axis=1).astype(np.float32)
    max_pop = np.max(item_pop) if item_pop.size else 1.0
    if max_pop == 0:
        max_pop = 1.0
    return tf.constant(item_pop[:num_items] / max_pop, dtype=tf.float32)


def build_training_data(user_item_matrix):
    user_items = {}
    user_pos_pairs = []
    neg_candidates = {}

    for user, row in enumerate(user_item_matrix):
        pos_items = np.flatnonzero(row)
        if pos_items.size == 0:
            continue

        neg_items = np.flatnonzero(1 - row)
        if neg_items.size == 0:
            continue

        user_items[user] = pos_items.tolist()
        neg_candidates[user] = neg_items
        for pos_item in pos_items:
            user_pos_pairs.append((user, pos_item))

    return user_items, user_pos_pairs, neg_candidates


def build_train_batches(batch_size, user_pos_pairs, neg_candidates, user_normals=None):
    if not user_pos_pairs:
        return []

    indices = np.random.permutation(len(user_pos_pairs))
    batches = []
    for start in range(0, len(user_pos_pairs), batch_size):
        end = min(start + batch_size, len(user_pos_pairs))
        batch_idx = indices[start:end]

        user_batch = []
        pos_batch = []
        neg_batch = []
        normal_mask = []
        for idx in batch_idx:
            user_id, pos_item = user_pos_pairs[idx]
            neg_item = np.random.choice(neg_candidates[user_id])
            user_batch.append(user_id)
            pos_batch.append(pos_item)
            neg_batch.append(neg_item)
            if user_normals is None:
                normal_mask.append(0.0)
            else:
                normal_mask.append(1.0 if pos_item in user_normals.get(user_id, set()) else 0.0)

        batches.append(
            (
                np.array(user_batch, dtype=np.int32),
                np.array(pos_batch, dtype=np.int32),
                np.array(neg_batch, dtype=np.int32),
                np.array(normal_mask, dtype=np.float32),
            )
        )

    return batches


class PropagatedEmbeddingModel(tf.keras.Model):
    def __init__(self, user_embedding, item_embedding, adj_norm, n_layers, num_users, num_items):
        super().__init__()
        self.user_embedding = user_embedding
        self.item_embedding = item_embedding
        self.adj_norm = adj_norm
        self.n_layers = n_layers
        self.num_users = num_users
        self.num_items = num_items

    def propagate_embeddings(self):
        u_emb0 = self.user_embedding(tf.range(self.num_users))
        i_emb0 = self.item_embedding(tf.range(self.num_items))
        all_emb0 = tf.concat([u_emb0, i_emb0], axis=0)

        embs = [all_emb0]
        for _ in range(self.n_layers):
            all_emb = tf.sparse.sparse_dense_matmul(self.adj_norm, embs[-1])
            embs.append(all_emb)
        all_emb = tf.reduce_mean(tf.stack(embs, axis=0), axis=0)
        return all_emb[:self.num_users], all_emb[self.num_users:]


class LightGCN(PropagatedEmbeddingModel):
    def __init__(self, num_users, num_items, embedding_dim, adj_norm, n_layers=3):
        user_embedding = tf.keras.layers.Embedding(num_users, embedding_dim, name='user_embeddings')
        item_embedding = tf.keras.layers.Embedding(num_items, embedding_dim, name='item_embeddings')
        super().__init__(user_embedding, item_embedding, adj_norm, n_layers, num_users, num_items)

    def call(self, user_idx, pos_item_idx, neg_item_idx):
        user_emb, item_emb = self.propagate_embeddings()
        u_emb_batch = tf.gather(user_emb, user_idx)
        pos_i_emb = tf.gather(item_emb, pos_item_idx)
        neg_j_emb = tf.gather(item_emb, neg_item_idx)
        x_uij = tf.reduce_sum(u_emb_batch * (pos_i_emb - neg_j_emb), axis=1)
        return -tf.reduce_mean(tf.math.log_sigmoid(x_uij))

    def get_embeddings(self):
        return self.propagate_embeddings()


class DCCLConf(PropagatedEmbeddingModel):
    def __init__(self, user_embedding, item_embedding, adj_norm, n_layers, num_users, num_items, item_pop):
        super().__init__(user_embedding, item_embedding, adj_norm, n_layers, num_users, num_items)
        self.item_pop = item_pop

    def call(self, user_idx, pos_item_idx, neg_item_idx):
        user_emb, item_emb = self.propagate_embeddings()
        u_emb_batch = tf.gather(user_emb, user_idx)
        pos_i_emb = tf.gather(item_emb, pos_item_idx)
        neg_j_emb = tf.gather(item_emb, neg_item_idx)
        pos_pred = tf.reduce_sum(u_emb_batch * pos_i_emb, axis=1)
        neg_pred = tf.reduce_sum(u_emb_batch * neg_j_emb, axis=1)
        pop = tf.clip_by_value(tf.gather(self.item_pop, pos_item_idx), 1e-6, 1.0)
        return -tf.math.log1p(-tf.exp(-pop)) + tf.math.softplus(neg_pred - pos_pred)


class DCCLInt(PropagatedEmbeddingModel):
    def __init__(self, user_embedding, item_embedding, adj_norm, n_layers, num_users, num_items, item_pop):
        super().__init__(user_embedding, item_embedding, adj_norm, n_layers, num_users, num_items)
        self.item_pop = item_pop

    def call(self, user_idx, pos_item_idx, neg_item_idx):
        user_emb, item_emb = self.propagate_embeddings()
        u_emb_batch = tf.gather(user_emb, user_idx)
        pos_i_emb = tf.gather(item_emb, pos_item_idx)
        neg_j_emb = tf.gather(item_emb, neg_item_idx)
        pos_pred = tf.reduce_sum(u_emb_batch * pos_i_emb, axis=1)
        neg_pred = tf.reduce_sum(u_emb_batch * neg_j_emb, axis=1)
        pop = tf.gather(self.item_pop, pos_item_idx)
        return pop + tf.math.softplus(neg_pred - pos_pred)


class LightGCNCombineLayer(tf.keras.Model):
    def __init__(self, e_u1, e_u2, e_i1, e_i2, adj_norm, n_layers, num_users, num_items):
        super().__init__()
        self.e_u1 = e_u1
        self.e_u2 = e_u2
        self.e_i1 = e_i1
        self.e_i2 = e_i2
        self.adj_norm = adj_norm
        self.n_layers = n_layers
        self.num_users = num_users
        self.num_items = num_items

    def _propagate(self, user_embedding, item_embedding):
        u_emb0 = user_embedding(tf.range(self.num_users))
        i_emb0 = item_embedding(tf.range(self.num_items))
        all_emb0 = tf.concat([u_emb0, i_emb0], axis=0)

        embs = [all_emb0]
        for _ in range(self.n_layers):
            all_emb = tf.sparse.sparse_dense_matmul(self.adj_norm, embs[-1])
            embs.append(all_emb)
        all_emb = tf.reduce_mean(tf.stack(embs, axis=0), axis=0)
        return all_emb[:self.num_users], all_emb[self.num_users:]

    def call(self, user_ids, pos_item_ids, neg_item_ids):
        u_emb_1, i_emb_1 = self._propagate(self.e_u1, self.e_i1)
        u_emb_2, i_emb_2 = self._propagate(self.e_u2, self.e_i2)

        user_latent1 = tf.gather(u_emb_1, user_ids)
        pos_item_latent1 = tf.gather(i_emb_1, pos_item_ids)
        neg_item_latent1 = tf.gather(i_emb_1, neg_item_ids)
        user_latent2 = tf.gather(u_emb_2, user_ids)
        pos_item_latent2 = tf.gather(i_emb_2, pos_item_ids)
        neg_item_latent2 = tf.gather(i_emb_2, neg_item_ids)

        user_latent = tf.concat([user_latent1, user_latent2], axis=1)
        pos_item_latent = tf.concat([pos_item_latent1, pos_item_latent2], axis=1)
        neg_item_latent = tf.concat([neg_item_latent1, neg_item_latent2], axis=1)

        pos_scores = tf.reduce_sum(user_latent * pos_item_latent, axis=-1)
        neg_scores = tf.reduce_sum(user_latent * neg_item_latent, axis=-1)
        return -tf.reduce_mean(tf.math.log_sigmoid(pos_scores - neg_scores))


def detect_normal_items(num_users, num_items, user_item_matrix, adj_norm, learning_rate, batch_size, embedding_dim, layers):
    lightgcn_model = LightGCN(num_users, num_items, embedding_dim, adj_norm, n_layers=layers)
    optimizer = create_optimizer(learning_rate)

    _, user_pos_pairs, neg_candidates = build_training_data(user_item_matrix)
    print('starting outlier detection...')
    for epoch in range(5):
        train_batches = build_train_batches(batch_size, user_pos_pairs, neg_candidates)
        for user_idx, pos_item_idx, neg_item_idx, _ in train_batches:
            with tf.GradientTape() as tape:
                loss = lightgcn_model(user_idx, pos_item_idx, neg_item_idx)

            gradients = tape.gradient(loss, lightgcn_model.trainable_variables)
            gradients_and_vars = [
                (grad, var)
                for grad, var in zip(gradients, lightgcn_model.trainable_variables)
                if grad is not None
            ]
            optimizer.apply_gradients(gradients_and_vars)

        print(f'Epoch {epoch + 1}/5, Loss: {loss.numpy():.6f}')

    user_embeddings, item_embeddings = lightgcn_model.get_embeddings()
    predicted_scores = np.dot(user_embeddings.numpy(), item_embeddings.numpy().T)

    k = max(1, num_items // 5)
    user_normals = {}
    for user_id, row in enumerate(user_item_matrix):
        score = predicted_scores[user_id, :]
        predicted_top_k_items = set(np.argsort(-score)[:k])
        true_items = set(np.flatnonzero(row))
        user_normals[user_id] = true_items.intersection(predicted_top_k_items)

    return user_normals


def train_model(data, embedding_dim, alpha, beta, layers, batch_size, learning_rate, epochs):
    num_users = data['num_users']
    num_items = data['num_items']
    train_a_mat = data['train_a1']

    adj_norm = to_sparse_tensor(train_a_mat)
    user_item_matrix = train_a_mat[:num_users, num_users:]
    item_pop_tensor = build_item_popularity(train_a_mat, num_users, num_items)

    user_items, user_pos_pairs, neg_candidates = build_training_data(user_item_matrix)
    if not user_items or not user_pos_pairs:
        raise RuntimeError('No valid user-item pairs were generated from the training data.')

    user_normals = detect_normal_items(
        num_users=num_users,
        num_items=num_items,
        user_item_matrix=user_item_matrix,
        adj_norm=adj_norm,
        learning_rate=learning_rate,
        batch_size=batch_size,
        embedding_dim=embedding_dim,
        layers=layers,
    )

    latent_dim = embedding_dim // 2
    e_u_pub = tf.keras.layers.Embedding(input_dim=num_users, output_dim=latent_dim)
    e_u_per = tf.keras.layers.Embedding(input_dim=num_users, output_dim=latent_dim)
    e_i_pop = tf.keras.layers.Embedding(input_dim=num_items, output_dim=latent_dim)
    e_i_nic = tf.keras.layers.Embedding(input_dim=num_items, output_dim=latent_dim)

    bpr_model_per_pop = DCCLInt(e_u_per, e_i_pop, adj_norm, layers, num_users, num_items, item_pop_tensor)
    bpr_model_pub_pop = DCCLConf(e_u_pub, e_i_pop, adj_norm, layers, num_users, num_items, item_pop_tensor)
    bpr_model_per_nic = DCCLInt(e_u_per, e_i_nic, adj_norm, layers, num_users, num_items, item_pop_tensor)
    bpr_model_pub_nic = DCCLConf(e_u_pub, e_i_nic, adj_norm, layers, num_users, num_items, item_pop_tensor)
    click_model = LightGCNCombineLayer(e_u_per, e_u_pub, e_i_pop, e_i_nic, adj_norm, layers, num_users, num_items)
    optimizer = create_optimizer(learning_rate)

    @tf.function
    def train_step(user_idx, pos_item_idx, neg_item_idx, normal_mask):
        with tf.GradientTape() as tape:
            ls_per_pop = bpr_model_per_pop(user_idx, pos_item_idx, neg_item_idx)
            ls_pub_pop = bpr_model_pub_pop(user_idx, pos_item_idx, neg_item_idx)
            ls_per_nic = bpr_model_per_nic(user_idx, pos_item_idx, neg_item_idx)
            ls_pub_nic = bpr_model_pub_nic(user_idx, pos_item_idx, neg_item_idx)

            per_mask = tf.cast(
                tf.gather(item_pop_tensor, pos_item_idx) >= tf.gather(item_pop_tensor, neg_item_idx),
                tf.float32,
            )

            per_pos_mask = per_mask * normal_mask
            pub_pos_mask = per_mask * (1.0 - normal_mask)
            per_neg_mask = (1.0 - per_mask) * normal_mask
            pub_neg_mask = (1.0 - per_mask) * (1.0 - normal_mask)

            l_per_val = tf.reduce_sum(ls_per_pop * per_pos_mask + ls_per_nic * per_neg_mask)
            l_pub_val = tf.reduce_sum(ls_pub_pop * pub_pos_mask + ls_pub_nic * pub_neg_mask)

            per_count = tf.reduce_sum(per_pos_mask + per_neg_mask)
            pub_count = tf.reduce_sum(pub_pos_mask + pub_neg_mask)

            zero = tf.constant(0.0, dtype=tf.float32)
            l_per_final = tf.cond(per_count > 0, lambda: l_per_val / per_count, lambda: zero)
            l_pub_final = tf.cond(pub_count > 0, lambda: l_pub_val / pub_count, lambda: zero)

            l_click = click_model(user_idx, pos_item_idx, neg_item_idx)
            loss = l_click + alpha * l_per_final + beta * l_pub_final

        gradients = tape.gradient(loss, click_model.trainable_variables)
        gradients_and_vars = [
            (grad, var)
            for grad, var in zip(gradients, click_model.trainable_variables)
            if grad is not None
        ]
        optimizer.apply_gradients(gradients_and_vars)
        return loss

    print('start model training...')
    for epoch in range(epochs):
        start_time = time.time()
        train_batches = build_train_batches(batch_size, user_pos_pairs, neg_candidates, user_normals)
        if not train_batches:
            raise RuntimeError('No training batches generated; check the dataset.')

        epoch_loss = 0.0
        for user_idx, pos_item_idx, neg_item_idx, normal_mask in train_batches:
            loss = train_step(user_idx, pos_item_idx, neg_item_idx, normal_mask)
            epoch_loss += float(loss.numpy())

        epoch_loss /= len(train_batches)
        elapsed = time.time() - start_time
        print(f'Epoch {epoch + 1}/{epochs}, Loss {epoch_loss:.6f}, Time {elapsed:.2f}s')

    return e_u_per, e_u_pub, e_i_pop, e_i_nic


def build_predicted_scores(e_u_per, e_u_pub, e_i_pop, e_i_nic):
    trained_user_embeddings = np.concatenate([e_u_per.get_weights()[0], e_u_pub.get_weights()[0]], axis=1)
    trained_item_embeddings = np.concatenate([e_i_pop.get_weights()[0], e_i_nic.get_weights()[0]], axis=1)
    return np.dot(trained_user_embeddings, trained_item_embeddings.T)


def evaluate_run(data, predicted_scores):
    num_users = data['num_users']
    test_a_mat = data['test_a1']
    test_a_mat_div = data['test_a1_div']

    print('\t{}\t{}'.format('\t'.join([f'Recall@{k}' for k in DEFAULT_KS]), '\t'.join([f'NDCG@{k}' for k in DEFAULT_KS])))

    test_adj_matrix = test_a_mat[:num_users, num_users:]
    recall_k, ndcg_k, per_user_hr, per_user_ndcg = evaluation.evaluate_model(test_adj_matrix, predicted_scores, DEFAULT_KS)
    print('random test\t{}\t{}'.format('\t'.join([f'{value:.5f}' for value in recall_k.values()]), '\t'.join([str(value) for value in ndcg_k.values()])))

    test_adj_matrix_div = test_a_mat_div[:num_users, num_users:]
    recall_k_div, ndcg_k_div, per_user_hr_div, per_user_ndcg_div = evaluation.evaluate_model(test_adj_matrix_div, predicted_scores, DEFAULT_KS)
    print('divers test\t{}\t{}'.format('\t'.join([f'{value:.5f}' for value in recall_k_div.values()]), '\t'.join([str(value) for value in ndcg_k_div.values()])))

    return {
        'acc': (recall_k, ndcg_k, recall_k_div, ndcg_k_div),
        'per_user': (per_user_hr, per_user_ndcg, per_user_hr_div, per_user_ndcg_div),
    }


def run_experiment(dataset, random_seed, embedding_dim, alpha, beta, layers, epochs, batch_size, learning_rate):
    set_random_seed(random_seed)
    tf.keras.backend.clear_session()

    print(f'load data with random seed: {random_seed}')
    data, default_batch_size, default_epochs, default_learning_rate = load_data.gen_data(dataset, random_seed)
    effective_batch_size = batch_size or default_batch_size
    effective_epochs = epochs or default_epochs
    effective_learning_rate = learning_rate or default_learning_rate

    num_users = data['num_users']
    num_items = data['num_items']
    test_a_mat_div = data['test_a1_div']
    non_zero_indices = np.argwhere(test_a_mat_div != 0)
    print('First 10 non-zero indices:')
    for idx in range(min(10, len(non_zero_indices))):
        print(non_zero_indices[idx])
    print('num_users:', num_users)
    print('num_items:', num_items)

    e_u_per, e_u_pub, e_i_pop, e_i_nic = train_model(
        data=data,
        embedding_dim=embedding_dim,
        alpha=alpha,
        beta=beta,
        layers=layers,
        batch_size=effective_batch_size,
        learning_rate=effective_learning_rate,
        epochs=effective_epochs,
    )

    predicted_scores = build_predicted_scores(e_u_per, e_u_pub, e_i_pop, e_i_nic)
    return evaluate_run(data, predicted_scores)


def summarize_results(results_by_seed):
    result_rows = []
    for result in results_by_seed.values():
        flattened = [value for metric_dict in result['acc'] for value in metric_dict.values()]
        result_rows.append(flattened)

    averages = np.mean(result_rows, axis=0)
    print('\t'.join(str(value) for value in averages))
    print(f'random test: NDCG@5\t{averages[5]:.5f}')
    print(f'intervened test: NDCG@5\t{averages[15]:.5f}')


def main():
    args = parse_args()
    output_path = Path(args.output) if args.output else SCRIPT_DIR / f'DSCP_LightGCN-{args.dataset}-results.npy'

    results_by_seed = {}
    for seed in args.random_seeds:
        print(f'Running seed {seed} on dataset {args.dataset}')
        results_by_seed[seed] = run_experiment(
            dataset=args.dataset,
            random_seed=seed,
            embedding_dim=args.embedding_dim,
            alpha=args.alpha,
            beta=args.beta,
            layers=args.layers,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )

    np.save(output_path, results_by_seed)
    print(f'Saved results to {output_path}')
    summarize_results(results_by_seed)


if __name__ == '__main__':
    main()