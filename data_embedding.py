import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from scipy.spatial.distance import cdist, pdist, squareform, euclidean
import ot


class Data_Embedding(object):
    def __init__(self, type):
        if type == 'product':
            self.model = Word2Vec.load("product2vector.model")

        self.vocab_len = len(self.model.wv.vocab)
        self.word2index = dict(zip([self.model.wv.index2word[i] for i in range(self.vocab_len)],
                              [i for i in range(self.vocab_len)]))
        self.word_index_df = pd.DataFrame(data=list(self.word2index.items()), columns=['product_id', 'emb_id'])

    def p2aisle_f(self, i):
        return self.p2aisles[i]

    def lookup_ind_f(self, i):
        return self.word2index[i]

        #Here we obtain the closest itemsets distance of cross vector network

    def obt_closest_of_set(self, item_id, set_of_candidates):
        vec_of_interest = self.model.wv.vectors[item_id]
        closest = np.argmin([euclidean(vec_of_interest, self.model.wv.vectors[x]) for x in set_of_candidates])
        return set_of_candidates[closest]

    def obt_closest_from_preds(self, pred, candidates_l_l):
        closest_from_history = []
        for p in pred:
            closest_from_history.append(self.obt_closest_of_set(p, [x for seq in candidates_l_l for x in seq]))
        return closest_from_history

    def basket_dist_LB(self, baskets):
        #Here we calculate the Lower Bound, which is actually a NN search.
        #We also find the similar dataset from basket 1 and basket 2 and following that will add all the distances which it finds minimum.
        basket1_vecs = self.model.wv.vectors[[x for x in baskets[0]]]
        basket2_vecs = self.model.wv.vectors[[x for x in baskets[1]]]

        dist_matrix = cdist(basket1_vecs, basket2_vecs)

        return max(np.mean(np.min(dist_matrix, axis=0)),
                   np.mean(np.min(dist_matrix, axis=1)))

    def basket_dist_EMD(self, baskets):
        basket1 = baskets[0]
        basket2 = baskets[1]
        dictionary = np.unique(list(basket1) + list(basket2))
        vocab_len_ = len(dictionary)
        product2ind = dict(zip(dictionary, np.arange(vocab_len_)))

        # Here distance matrix is calculated.
        dict_vectors = self.model.wv.vectors[[x for x in dictionary]]
        dist_matrix = squareform(pdist(dict_vectors))

        if np.sum(dist_matrix) == 0.0:
            # There will be issues if 'EMD' has 0s in it.
            return float('inf')

        def no_bow(document):
            bow = np.zeros(vocab_len_, dtype=np.float)
            for d in document:
                bow[product2ind[d]] += 1.
            return bow / len(document)

        # 'no_bow' is alculated to represent data as documets.
        d1 = no_bow(basket1)
        d2 = no_bow(basket2)

        # Here we obtain Wasserstein Minimim Distance.
        return ot.emd2(d1, d2, dist_matrix)

    def rmv_prod_wo_emb(self, all_baskets):
        final_filtered_baskets = []
        for s in all_baskets:
            s_cp = []
            for b in s:
                b_cp = [x for x in b if x in self.model.wv.vocab]
                if len(b_cp) > 0:
                    s_cp.append(b_cp)
            if len(s_cp) > 0:
                final_filtered_baskets.append(s_cp)
        return final_filtered_baskets
