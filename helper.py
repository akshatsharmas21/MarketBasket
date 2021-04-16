from sklearn.model_selection import train_test_split
from collections import Counter


def nest_chng(item, func):
    if isinstance(item, list):
        return [nest_chng(x, func) for x in item]
    return func(item)

# Here we are deleting the itemsets which are hardly considered.

def rm_diff_products(all_baskets, max_num=500):
    print('Removing all but {} most common products'.format(max_num))
    p = []
    for a in all_baskets:
        for b in a:
            p.extend(b)
    prod_contr = Counter(p)
    most_used_prodts = [x for x, _ in prod_contr.most_common(max_num)]
    final_filtered_baskets = []
    for a in all_baskets:
        a__cp = []
        for c in a:
            c__cp = [x for x in c if x in most_used_prodts]
            if len(c__cp) > 0:
                a__cp.append(c__cp)
        if len(a__cp) > 0:
            final_filtered_baskets.append(a__cp)
    return final_filtered_baskets


def rm_small_baskets(all_baskets, l_b = 5, l_s = 10):
    final_filtered_baskets = []
    for s in all_baskets:
        s__cp = []
        for b in s:
            if len(b) > l_b:
                s__cp.append(b)
        if len(s__cp) > l_s:
            final_filtered_baskets.append(s__cp)
    return final_filtered_baskets


def div_data(all_baskets):
    train_upbd, test_upbd = train_test_split(all_baskets, test_size=0.05, random_state=0)
    train_upbd, val_ub = train_test_split(train_upbd, test_size=0.05, random_state=0)

    test_upbd_input = [x[:-1] for x in test_upbd]
    test_upbd_target = [x[-1] for x in test_upbd]

    val_ub_input = [x[:-1] for x in val_ub]
    val_ub_target = [x[-1] for x in val_ub]

    return train_upbd, val_ub_input, val_ub_target, test_upbd_input, test_upbd_target
