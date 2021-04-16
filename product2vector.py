import pandas as pnds
import gensim

path_train = "/content/code/order_products__train.csv"
path_prior = "/content/code/order_products__prior.csv"
path_products = "/content/code/products.csv"

train_orders = pnds.read_csv(path_train)
prior_orders = pnds.read_csv(path_prior)
products = pnds.read_csv(path_products)

# Here we covert product ID to more readable format i.e., string
train_orders["product_id"] = train_orders["product_id"].astype(str)
prior_orders["product_id"] = prior_orders["product_id"].astype(str)

train_products = train_orders.groupby("order_id").apply(lambda order: order['product_id'].tolist())
prior_products = prior_orders.groupby("order_id").apply(lambda order: order['product_id'].tolist())

#Here we make the final sentence that combines datasets from train and prior
sntnc = prior_products.append(train_products).values

# Here we train a model based on the readings obtained from sentence.
model = gensim.models.Word2Vec(sntnc, size=50, window=5, min_count=50, workers=4)

model.save("product2vector.model")
model.wv.save_word2vec_format("product2vector.model.bin", binary=True)
