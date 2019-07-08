# coding: utf-8
import argparse
import pandas as pd
import os

parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM classification Model')
parser.add_argument('--data', type=str, default=os.getcwd()+'/ag_news_csv/',
                    help='location of the data corpus')
parser.add_argument('--number_per_class', type=int, default=1000,
                    help='location of the data corpus')

args = parser.parse_args()
train_file_name = os.path.join(args.data, 'train.csv')
train_data = pd.read_csv(train_file_name, header=None)
# the labeled data are set to be the top "number_per_class" rows of each group
labeled_data = train_data.groupby(train_data[0]).head(args.number_per_class)
unlabeled_data = train_data.drop(labeled_data.index)
# save data to labeled and unlabeled data separately
labeled_data.to_csv(os.path.join(args.data, str(args.number_per_class)+'_labeled_train.csv'), header=False, index=False)
unlabeled_data.to_csv(os.path.join(args.data, str(args.number_per_class)+'_unlabeled_train.csv'), header=False, index=False)
