import pandas
import sklearn.metrics
import sklearn.linear_model
import sklearn.neighbors
import sklearn.tree
import sklearn.model_selection
import re
from scipy.stats import ttest_ind
from sklearn.model_selection import cross_val_score

unique_data = pandas.read_csv('my_data.txt', sep = "\t")

def clean_data(frame):
    
    def caps(words):
        return ' '.join([word.strip().capitalize() for word in words.split(' ')])
        
    frame.dropna(inplace=True)
    ignore_columns = ['label', 'col_02', 'col_03', 'col_07', 'col_12']
    if 'col_07' in frame.columns:
        frame['col_07'] = pandas.to_numeric(frame['col_07'], errors='coerce').astype('Int64')
    if 'col_12' in frame.columns:
        frame['col_12'] = pandas.to_numeric(frame['col_12'], errors='coerce').astype('Int64')
    for column in frame.columns:
        if column not in ignore_columns:
            for index, row in frame[column].items():
                find = re.findall(r'[\d.,%-]+', str(row))
                if find:
                    frame.at[index, column] = find[0]
            frame[column] = pandas.to_numeric(frame[column], errors='coerce')
    if 'col_02' in frame.columns:
        frame['col_02'] = frame['col_02'].apply(caps)
    if 'col_03' in frame.columns:
        frame['col_03'] = frame['col_03'].apply(caps)
    return frame
    
unique_data = clean_data(unique_data)
unique_data