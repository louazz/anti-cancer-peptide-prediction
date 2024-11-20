import pandas as pd
import peptides

active = pd.read_csv('/home/louai/Documents/anticancer_peptide/dataset/AntiCP/active.csv')
active['class']= 1

inactive = pd.read_csv('/home/louai/Documents/anticancer_peptide/dataset/AntiCP/inactive.csv')
inactive['class']= 0

df = pd.concat([active, inactive])
label = df['class']
print(df.head)
descriptors = []
for index, row in df.iterrows():
    sequence = row['sequence']
    desc = peptides.Peptide(sequence).descriptors()
    desc['class']= row['class']
    descriptors.append(desc)

desc = pd.DataFrame(descriptors)

df = desc.sample(frac=1)
df.to_csv('descriptors.csv', sep=',')