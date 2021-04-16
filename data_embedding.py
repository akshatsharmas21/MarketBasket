import pandas as pnds
import numpy as nmpy
import ot
from scipy.spatial.distance import euclidean, pdist, cdist, squareform
from gensim.models import Word2Vec


class Data_Embedding(object):
  def __init__(self, type):
    if type == 'product':
      self.model = Word2Vec.load("product2vector.model")

    self.vocab_len = len(self.model.wv.vocab)
    self.w2i = dict(zip([self.model.wv.index2word[i] for i in range(self.vocab_len)],
            [i for i in range(self.vocab_len)]))
    self.word_index_df = pnds.DataFrame(data=list(self.w2i.items()), columns=['product_id', 'emb_id'])


  def lkup(self, i):
    return self.w2i[i]

#Here we obtain the closest itemsets distance of cross vector network

  def obt_closest_of_set(self, itm_id, st_cand):
    vector_intrst = self.model.wv.vectors[itm_id]
    clst = nmpy.argmin([euclidean(vector_intrst, self.model.wv.vectors[x]) for x in st_cand])
    return st_cand[clst]

  def obt_closest_from_preds(self, pred, cand_l_of_l):
    cls_hist = []
    for p in pred:
      cls_hist.append(self.obt_closest_of_set(p, [x for seq in cand_l_of_l for x in seq]))
    return cls_hist

  def basket_dist_LB(self, bskts):
#Here we calculate the Lower Bound, which is actually a NN search.
#We also find the similar dataset from basket 1 and basket 2 and following that will add all the distances which it finds minimum.
    bsk1_vectrs = self.model.wv.vectors[[x for x in bskts[0]]]
    bsk2_vectrs = self.model.wv.vectors[[x for x in bskts[1]]]

    dist_matrix = cdist(bsk1_vectrs, bsk2_vectrs)

    return max(nmpy.mean(nmpy.min(dist_matrix, axis=0)),
      nmpy.mean(nmpy.min(dist_matrix, axis=1)))

  def basket_dist_Decomp(self, bskts):
    bskt1 = bskts[0]
    bskt2 = bskts[1]
    dct = nmpy.unique(list(bskt1) + list(bskt2))
    vocab_len_ = len(dct)
    product2ind = dict(zip(dct, nmpy.arange(vocab_len_)))

# Here distance matrix is calculated.
    dict_vectors = self.model.wv.vectors[[x for x in dct]]
    dist_matrix = squareform(pdist(dict_vectors))

    if nmpy.sum(dist_matrix) == 0.0:
# There will be issues if 'EMD' has 0s in it.
      return float('inf')

    def no_bow(doc):
      bow = nmpy.zeros(vocab_len_, dtype=nmpy.float)
      for e in doc:
        bow[product2ind[e]] += 1.
      return bow / len(doc)

# 'no_bow' is calculated to represent data as documets.
    dist_1 = no_bow(bskt1)
    dist_2 = no_bow(bskt2)

# Here we obtain Wasserstein Minimim Distance.
    return ot.emd2(dist_1, dist_2, dist_matrix)

  def rmv_prod_wo_emb(self, all_baskets):
    final_filtered_baskets = []
    for p in all_baskets:
      p__cp = []
      for q in p:
        q__cp = [x for x in q if x in self.model.wv.vocab]
        if len(q__cp) > 0:
          p__cp.append(q__cp)
      if len(p__cp) > 0:
        final_filtered_baskets.append(p__cp)
    return final_filtered_baskets
