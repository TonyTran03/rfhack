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

# next steps

rough idea: we know if $f: A\rightarrow B$ and $g: B\rightarrow C$ are statistical functions, and both $f$ and $g$ are provably surjective, then for any result $c\in C$, there should exist $a\in A$ such that $g(f(a))=c$, i.e., that surjective maps are closed under composition. 

Concretely, for any metric $m$ or set of metrics $M=\{m_1,m_2,\ldots\}$ with their metric functions $f$ known beforehand, if we can prove surjectivity, then for any $m$ or $M$, there should be an input in the preimage that, when evaluated by $f$, will produce itself. Alternatively, if we cannot prove surjectivity onto the codomain, can we precisely describe one of its subsets that is surjective?
