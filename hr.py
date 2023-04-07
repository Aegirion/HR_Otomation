import numpy as np
import pandas as pd
from sklearn import tree

df = pd.read_csv("dtd.csv")
df.head()

dztlm_mapping = {'Y': 1, 'N': 0}
df['IseAlindi'] = df['IseAlindi'].map(dztlm_mapping)
df['SuanCalisiyor?'] = df['SuanCalisiyor?'].map(dztlm_mapping)
df['Top10 Universite?'] = df['Top10 Universite?'].map(dztlm_mapping)
df['StajBizdeYaptimi?'] = df['StajBizdeYaptimi?'].map(dztlm_mapping)

dztlm_mapping_egitim = {'BS': 0, 'MS': 1, 'PhD': 2}
df['Egitim Seviyesi'] = df['Egitim Seviyesi'].map(dztlm_mapping_egitim)
df.head()

y = df["IseAlindi"]
x = df.drop(["IseAlindi"], axis=1)

x = x.fillna(0)

train = tree.DecisionTreeClassifier()
train = train.fit(x, y)

a = int(input("Kaç yıl deneyiminiz var?: "))
b = int(input("Şu an çalışıyor musunuz? (0-Hayır, 1-Evet): "))
c = int(input("Önceden kaç firmada çalıştınız?: "))
d = int(input("Eğitim seviyeniz nedir? (0-Lisans, 1-Yüksek Lisans, 2-Master): "))
e = int(input("En iyi 10 üniversiteden birinde mi okudunuz? (0-Hayır, 1-Evet): "))
f = int(input("Stajınızı bizde mi yaptınız? (0-Hayır, 1-Evet): "))

m = train.predict([[a, b, c, d, e, f]])

if m[0] == 1:
print("İşe alındınız.")
else:
print("Kriterleri karşılamadınız.")