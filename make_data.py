import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json

excel_file = 'QA_dataset_v9_20210218_975.xlsx'
df = pd.read_excel(excel_file, header=0,engine='openpyxl')

inten = df[['question','intent']].dropna()
inten = inten.rename(columns={'question':'titletext','intent':'label'})
inten = inten[['label','titletext']]
dic = {k:n for n,k in enumerate(set(inten['label']))}
inten = inten.replace(dic)

with open('inten_dic.json', 'w') as outfile:
    json.dump(dic, outfile)
    
depart = df[['question','department']].dropna()
depart = depart.rename(columns={'question':'titletext','department':'label'})
depart = depart[['label','titletext']]
dic = {k:n for n,k in enumerate(set(depart['label']))}
depart = depart.replace(dic)

with open('depart_dic.json', 'w') as outfile:
    json.dump(dic, outfile)
    
from sklearn.model_selection import train_test_split
inten_train,inten_valid = train_test_split(inten, test_size=0.5,shuffle =True ,random_state=42)
inten_train.to_csv('data/inten_train.csv', index=False)
inten_valid.to_csv('data/inten_valid.csv', index=False)

depart_train,depart_valid = train_test_split(depart, test_size=0.5,shuffle =True ,random_state=42)
depart_train.to_csv('data/depart_train.csv', index=False)
depart_valid.to_csv('data/depart_valid.csv', index=False)
