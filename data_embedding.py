import os
import pickle
import pandas as pd


class BuildingConstructor(object):
    '''
        Grouping of products in their respective baskets in the form of listings
    '''
    def __init__(self, unch_dir, che_dir):
        self.unch_dir = unch_dir
        self.che_dir = che_dir

    def obt_orders(self):
        '''
            getting only the relevant column names from the datasets which the algorithm will work on
        '''
        orders = pd.read_csv(self.unch_dir + 'orders.csv')
        orders = orders.fillna(0.0)
        orders['days'] = orders.groupby(['user_id'])['days_since_prior_order'].cumsum()
        orders['days_last'] = orders.groupby(['user_id'])['days'].transform(max)
        orders['days_up_to_last'] = orders['days_last'] - orders['days']
        del orders['days_last']
        del orders['days']
        return orders

    def obt_orders_items(self, prior_or_train):
        '''
            obtaining the complete data via prior or train orders
        '''
        orders_products = pd.read_csv(self.unch_dir + 'order_products__%s.csv'%prior_or_train)
        return orders_products

    def obt_users_orders(self, prior_or_train):
        '''
            obtaining the prior information of the users
        '''
        if os.path.exists(self.che_dir + 'users_orders.pkl'):
            with open(self.che_dir + 'users_orders.pkl', 'rb') as f:
                usr_ordr = pickle.load(f)
        else:
            orders = self.obt_orders()
            order_products_prior = self.obt_orders_items(prior_or_train)
            usr_ordr = pd.merge(order_products_prior, orders[['user_id', 'order_id', 'order_number', 'days_up_to_last']],
                        on = ['order_id'], how = 'left')
            with open(self.che_dir + 'users_orders.pkl', 'wb') as f:
                pickle.dump(usr_ordr, f, pickle.HIGHEST_PROTOCOL)
        return usr_ordr

    def obt_users_products(self, prior_or_train):
        '''
            obtain the overall data for the users brought products
        '''
        if os.path.exists(self.che_dir + 'users_products.pkl'):
            with open(self.che_dir + 'users_products.pkl', 'rb') as f:
                usr_prod = pickle.load(f)
        else:
            usr_prod = self.obt_users_orders(prior_or_train)[['user_id', 'product_id']].drop_duplicates()
            usr_prod['product_id'] = usr_prod.product_id.astype(int)
            usr_prod['user_id'] = usr_prod.user_id.astype(int)
            usr_prod = usr_prod.groupby(['user_id'])['product_id'].apply(list).reset_index()
            with open(self.che_dir + 'users_products.pkl', 'wb') as f:
                pickle.dump(usr_prod, f, pickle.HIGHEST_PROTOCOL)
        return usr_prod

    def obt_items(self, gran):
        '''
            obtain the necessary items' details
        '''
        items = pd.read_csv(self.unch_dir + '%s.csv'%gran)
        return items

    def get_baskets(self, prior_or_train, reconstruct = False, none_idx = 49689):
        '''
            obtain the overall basket of the users
        '''
        f_p = self.che_dir + './basket_' + prior_or_train + '.pkl'

        if os.path.exists(f_p):
            with open(f_p, 'rb') as f:
                ubskt = pickle.load(f)
        else:
            up = self.obt_users_orders(prior_or_train).sort_values(['user_id', 'order_number', 'product_id'], ascending = True)
            uid_oid = up[['user_id', 'order_number']].drop_duplicates()
            up = up[['user_id', 'order_number', 'product_id']]
            ubskt = up.groupby(['user_id', 'order_number'])['product_id'].apply(list).reset_index()
            ubskt = pd.merge(uid_oid, ubskt, on = ['user_id', 'order_number'], how = 'left')
            for row in ubskt.loc[ubskt.product_id.isnull(), 'product_id'].index:
                ubskt.at[row, 'product_id'] = [none_idx]
            ubskt = ubskt.sort_values(['user_id', 'order_number'], ascending = True).groupby(['user_id'])['product_id'].apply(list).reset_index()
            ubskt.columns = ['user_id', 'basket']
            with open(f_p, 'wb') as f:
                pickle.dump(ubskt, f, pickle.HIGHEST_PROTOCOL)
        return ubskt

    def get_item_history(self, prior_or_train, reconstruct = False, none_idx = 49689):
        f_p = self.che_dir + './item_history_' + prior_or_train + '.pkl'
        if (not reconstruct) and os.path.exists(f_p):
            with open(f_p, 'rb') as f:
                itm_hist = pickle.load(f)
        else:
            up = self.obt_users_orders(prior_or_train).sort_values(['user_id', 'order_number', 'product_id'], ascending = True)
            itm_hist = up.groupby(['user_id', 'order_number'])['product_id'].apply(list).reset_index()
            itm_hist.loc[itm_hist.order_number == 1, 'product_id'] = itm_hist.loc[itm_hist.order_number == 1, 'product_id'] + [none_idx]
            itm_hist = itm_hist.sort_values(['user_id', 'order_number'], ascending = True)
            # Gather datasets by taking user id and product id.
            itm_hist['product_id'] = itm_hist.groupby(['user_id'])['product_id'].transform(pd.Series.cumsum)
            # Obtaining distictive data set here.
            itm_hist['product_id'] = itm_hist['product_id'].apply(set).apply(list)
            itm_hist = itm_hist.sort_values(['user_id', 'order_number'], ascending = True)
            # Moving individual category and putting them into history set.
            itm_hist['product_id'] = itm_hist.groupby(['user_id'])['product_id'].shift(1)
            for row in itm_hist.loc[itm_hist.product_id.isnull(), 'product_id'].index:
                itm_hist.at[row, 'product_id'] = [none_idx]
            itm_hist = itm_hist.sort_values(['user_id', 'order_number'], ascending = True).groupby(['user_id'])['product_id'].apply(list).reset_index()
            itm_hist.columns = ['user_id', 'history_items']

            with open(f_p, 'wb') as f:
                pickle.dump(itm_hist, f, pickle.HIGHEST_PROTOCOL)
        return itm_hist
