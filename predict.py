import argparse
import numpy as np
import pandas as pd
from utils import final_feature_matrix, predict
import pickle

# Parsing script arguments
parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('tsv_path', type=str, help='tsv file path')
args = parser.parse_args()

# Reading input TSV
data = pd.read_csv(args.tsv_path, sep="\t")

#####
# prediction code
with open("model.pkl", 'rb') as file:
    model = pickle.load(file)
X_test = final_feature_matrix(data)
prediction_df = pd.DataFrame(columns=['id', 'revenue'])
prediction_df['id'] = data['id']
prediction_df['revenue'] = predict(model,X_test)
####

# export prediction results
prediction_df.to_csv("prediction.csv", index=False, header=False)