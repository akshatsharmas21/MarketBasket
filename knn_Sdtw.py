import numpy as np
from collections import Counter
from data_embedding import Data_Embedding
from tqdm import tqdm

class Knn_Sdtw(object):
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors
        self.len_to_take = 10

    def _sdtw_dist(self, ts_x, ts_y, best_for_ts_x, d, d_lower_bound):

        # Here we compute cost matrix with large integer
        M, N = len(ts_x), len(ts_y)

        # Here we calculate Lower Bound distances
        LB_gen = map(d_lower_bound, [(i,j) for i in ts_x for j in ts_y])
        d_LB_min = np.fromiter(LB_gen, dtype=np.float)

        #The code will come out of the loop of there is no shorter option available
        if np.sum(d_LB_min[np.argpartition(d_LB_min, M)][:M]) > max(best_for_ts_x):
            return np.inf, ts_y[0]

        cost = np.inf * np.ones((M, N))

        #Calculate the overall distances
        d_matrix = np.zeros((M,N))
        for i in range(M):
            for j in range(N):
                d_matrix[i,j] = d((ts_x[i], ts_y[j]))

        # Here Initialization of first row and column is done.
        cost[0, 0] = d((ts_x[0], ts_y[0]))
        for i in range(1, M):
            cost[i, 0] = cost[i-1, 0] + d_matrix[i, 0]

        for j in range(1, N):
            cost[0, j] = d_matrix[0, j]

        # Leftover cost matrix ate populated here
        for i in range(1, M):
            w = 1.
            for j in range(1, N):
                choices = cost[i-1, j-1], cost[i, j-1], cost[i-1, j]
                cost[i, j] = min(choices) + w * d_matrix[i,j]

        min_index = np.argmin(cost[-1,:-1])
        # Here Dynamic Time Warping is returned for next basket prediction
        return cost[-1,min_index], ts_y[min_index + 1]

    def _dist_matrix(self, x, y, d, d_lower_bound):
        # Here we calculate overall distance matrix between x and y
        x_s = np.shape(x)
        y_s = np.shape(y)
        dm = np.inf * np.ones((x_s[0], y_s[0]))
        nxt_baskets = np.empty((x_s[0], y_s[0]), dtype=object)

        for i in tqdm(range(0, x_s[0])):
            x[i] = np.array(x[i])
            if x[i].shape[0] > self.len_to_take:
                x[i] = x[i][-self.len_to_take:]
            best_dist = [np.inf] * max(self.n_neighbors)
            for j in range(0, y_s[0]):
                y[j] = np.array(y[j])
                dist, pred = self._sdtw_dist(x[i], y[j], best_dist, d, d_lower_bound)
                if dist < np.max(best_dist):
                    best_dist[np.argmax(best_dist)] = dist
                dm[i, j] = dist
                nxt_baskets[i, j] = pred

        return dm, nxt_baskets

    def predict(self, tr_d, te_d, d, d_lower_bound):
        dm, predictions = self._dist_matrix(te_d, tr_d, d, d_lower_bound)

        preds_total_l = []
        distances_total_l = []
        for k in self.n_neighbors:
            # Here we recognize the knn
            knn_idx = dm.argsort()[:, :k]
            preds_k_l = []
            distances_k_l = []

            for i in range(len(te_d)):
                preds = [predictions[i][knn_idx[i][x]] for x in range(knn_idx.shape[1])]
                distances = np.mean([dm[i][knn_idx[i][x]] for x in range(knn_idx.shape[1])])
                pred_len = int(np.mean([len(te_d[i][x]) for x in range(len(te_d[i]))]))
                preds = [x for x, y in Counter([n for s in preds for n in s]).most_common(pred_len)]
                preds_k_l.append(preds)
                distances_k_l.append(distances)
            preds_total_l.append(preds_k_l)
            distances_total_l.append(distances_k_l)

        return preds_total_l, distances_total_l
