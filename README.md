# MarketBasket
This implementation is based on the concepts, ideas and algorithm learned from 'Personalized Purchase Prediction of Market Baskets with Wasserstein-Based Sequence Matching', accepted for oral presentation at 25th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD 2019).

# Running Procedure
1. We used Google Colab as it will make the proceding smoother (less taxing on our system's underpowered hardware).
2. Import the git repository as well as Kaggle's datasets by unzipping them in the current working directory.
3. Kaggle's dataset can be imported in the working directory by using 'json' file obtained from our Kaggle account's profile. We can also upload the datasets normally by uploading the data ourselves.
4. After this we have to run 'product2vector.py' file for generating product embeddings.
5. Lastly we have to run 'main.py' to compute everything and getting the desired prediction.

# References
@inproceedings{Kraus:2019:PPP:3292500.3330791,
 author = {Kraus, Mathias and Feuerriegel, Stefan},
 title = {Personalized Purchase Prediction of Market Baskets with Wasserstein-Based Sequence Matching},
 booktitle = {Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining},
 series = {KDD '19},
 year = {2019},
 doi = {10.1145/3292500.3330791},
 } 
