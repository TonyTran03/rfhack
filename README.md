Pedagogical experiment demonstrating exploitability of area under a receiver operating characteristic curve $\text{A}$ in random forest classifiers. 

Given any sufficiently sized dataset $D$, and a value $0.5+\epsilon \leq k\leq 1-\epsilon$, we can produce $D'$ such that for any valid split of $D\cup D'$ used to train a random forest, testing accuracy on held out data points for classification tasks will have $A\approx k$.  

# usage

```python
h = rfhack.AdversarialHacker(df)
h.hack(n)

h.df # final dataset with downstream AUC of approximately n
h.auc # average AUC
h.min # minimum AUC across iterations
h.max # maximum AUC across iterations
```

