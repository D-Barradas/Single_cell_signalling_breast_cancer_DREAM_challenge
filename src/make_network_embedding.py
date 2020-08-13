import pandas as pd, matplotlib.pyplot as plt
import networkx as nx
from node2vec import Node2Vec
plt.ion()
#inttable = pd.read_table('string_results/string_interactions_network.tsv')
#graph = nx.from_pandas_edgelist(inttable,'#node1','node2')
inttable = pd.read_table('prior_knowledge.sif', header=None, names=['n1','rel','n2'])
graph = nx.from_pandas_edgelist(inttable,'n1','n2')
node2vec = Node2Vec(graph, dimensions=16, walk_length=8, num_walks=200, workers=4)
model = node2vec.fit(window=10, min_count=1, batch_words=4)
model.save('network_embedding.mod')
model.wv.save_word2vec_format('network_embedding.emb')

# make network drawing
refnode = 'mTOR'
refsimils = dict(model.wv.most_similar(refnode,topn=len(model.wv.vocab))+[(refnode,1)])
fig, ax = plt.subplots()
nx.draw_networkx(graph, node_color=[refsimils[n] for n in graph.nodes])
ax.set_title('Color in reference to node '+refnode)
fig.savefig('network_embedding.svg')
