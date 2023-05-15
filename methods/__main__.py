import permanence

additionality = [ 1.0, 1.1 ]
leakage = [ 0.5, 0.6 ]

c = permanence.net_sequestration(additionality, leakage, 1)

print(c)
