import pandas as pd
import peptides
from imblearn.under_sampling import RandomUnderSampler

path = '/home/louai/Documents/anticancer_peptide/dataset/ACPs_Breast_cancer.csv'

data = pd.read_csv(path, sep=',')
features = data[['sequence', 'class']]

mapping = {'mod. active': 1, 'very active': 1, 'inactive - exp': 0, 'inactive - virtual': 0}

df = data.replace({'class': mapping})
label = df['class']

descriptors = []
for index, row in df.iterrows():
    sequence = row['sequence']
    desc = peptides.Peptide(sequence).descriptors()
    descriptors.append(desc)

df = pd.DataFrame(descriptors)
df['class'] = label

sampler = RandomUnderSampler(sampling_strategy='majority')

y = df['class']
X = df.drop(columns=['class'])

X, y = sampler.fit_resample(X, y)
df = X
df['class'] = y

print(df['class'].value_counts())

df.to_csv('descriptors.csv', sep=',')
