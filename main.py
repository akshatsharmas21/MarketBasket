from data import BuildingConstructor
from knn_Sdtw import Knn_Sdtw
from data_embedding import Data_Embedding
from helper import nest_chng, rm_diff_products, rm_small_baskets, div_data


def run():
    data_embedding = Data_Embedding('product')
    bc = BuildingConstructor('/content/code/', '/content/code/')
    ub_basket = bc.get_baskets('prior', reconstruct=False)

    all_baskets = ub_basket.basket.values
    all_baskets = nest_chng(list(all_baskets), str)

    all_baskets = data_embedding.rmv_prod_wo_emb(all_baskets)
    all_baskets = rm_diff_products(all_baskets)
    all_baskets = rm_small_baskets(all_baskets)
    all_baskets = nest_chng(all_baskets, data_embedding.lookup_ind_f)

    train_upbd, val_ub_input, val_ub_target, test_upbd_input, test_upbd_target = div_data(all_baskets)

    knn_Sdtw = Knn_Sdtw(n_neighbors=[5])
    preds_all, distances = knn_Sdtw.predict(train_upbd, val_ub_input, data_embedding.basket_dist_EMD,
                                          data_embedding.basket_dist_LB)
    return preds_all, distances


if __name__ == "__main__":
    run()
