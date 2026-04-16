import numpy as np
import tensorflow as tf

def evaluate_model(test_adj_matrix, predicted_scores, ks):
    """
    Evaluate Recall@K and NDCG@K for multiple values of K.
    
    Args:
        test_adj_matrix (np.ndarray): Binary matrix of ground truth interactions (shape: num_users x num_items).
        predicted_scores (np.ndarray): Predicted scores for all user-item pairs (shape: num_users x num_items).
        ks (list of int): List of K values to compute Recall@K and NDCG@K.
    
    Returns:
        recall_k (dict): Dictionary where keys are K and values are Recall@K.
        ndcg_k (dict): Dictionary where keys are K and values are NDCG@K.
    """
    def dcg_at_k(rel_scores, k):
        """Compute the Discounted Cumulative Gain (DCG) at K."""
        rel_scores = np.asfarray(rel_scores)[:k]
        if rel_scores.size == 0:
            return 0.0
        return np.sum(rel_scores / np.log2(np.arange(2, rel_scores.size + 2)))
    
    def ideal_dcg_at_k(rel_scores, k):
        """Compute the Ideal Discounted Cumulative Gain (IDCG) at K."""
        sorted_rel_scores = sorted(rel_scores, reverse=True)
        return dcg_at_k(sorted_rel_scores, k)

    num_users, num_items = test_adj_matrix.shape
    recalls = {k: [] for k in ks}
    ndcgs = {k: [] for k in ks}

    per_user_hr = {}
    per_user_ndcg = {}
    for user in range(num_users):
        # Get true items the user interacted with
        true_items = np.where(test_adj_matrix[user] == 1)[0]
        
        if len(true_items) == 0:
            continue  # Skip users with no interactions in the test set
        
        # Get the predicted ranking of items
        top_items = np.argsort(-predicted_scores[user])

        per_user_hr[user] = {}
        per_user_ndcg[user] = {}
        for k in ks:    
            # Top K items from the predicted scores
            top_k_items = top_items[:k]

            # Compute Recall@K
            num_relevant = len(set(top_k_items) & set(true_items))
            recall = num_relevant / len(true_items)
            recalls[k].append(recall)
            per_user_hr[user][k] = recall

            # Compute NDCG@K
            rel_scores = [1 if item in true_items else 0 for item in top_k_items]
            dcg = dcg_at_k(rel_scores, k)
            idcg = ideal_dcg_at_k([1] * len(true_items), k)
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcgs[k].append(ndcg)
            per_user_ndcg[user][k] = ndcg

    # Compute average Recall@K and NDCG@K for each K
    recall_k = {k: np.mean(recalls[k]) for k in ks}
    ndcg_k = {k: np.mean(ndcgs[k]) for k in ks}
    
    return recall_k, ndcg_k, per_user_hr, per_user_ndcg

def predict_interaction_matrix(model, n_users, n_items, batch_size=512):
    Y_hat = np.zeros((n_users, n_items), dtype=np.float32)

    for u in range(n_users):
        # prepare user id tensor
        u_tensor = tf.fill([batch_size], u)  # we will reuse this in batches

        scores_user = []
        for start in range(0, n_items, batch_size):
            end = min(start + batch_size, n_items)
            i_batch = tf.range(start, end, dtype=tf.int32)
            u_batch = tf.fill([end - start], u)

            s_batch, _, _, _ = model.score(u_batch, i_batch)
            scores_user.append(s_batch.numpy().reshape(-1))

        scores_user = np.concatenate(scores_user, axis=0)
        Y_hat[u, :] = scores_user

    return Y_hat