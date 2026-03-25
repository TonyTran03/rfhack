import numpy
import pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from rfhack import AdversarialHacker
 
numpy.random.seed(20050119)
 

real = pandas.read_csv("missForest.csv")
 
hacker = AdversarialHacker(real)
result = hacker.hack(0.7)

 

fake = result.df
 
real_labeled = real.copy()
real_labeled['target'] = 1
fake_labeled = fake.copy()
fake_labeled['target'] = 0
combined = pandas.concat([real_labeled, fake_labeled], ignore_index=True)
 
X = combined.drop(columns=['target'])
y = combined['target']
 

for i in range(10):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=i)
    clf = RandomForestClassifier(n_estimators=100, random_state=i)
    clf.fit(X_tr, y_tr)
    probs = clf.predict_proba(X_te)[:, 1]
    print(f"roc ={roc_auc_score(y_te, probs):.4f}")