'''Reads the graph and node contexts'''

import networkx as nx

def build_graph():
    '''Read input network''' 

    G = nx.read_edgelist('input/cora.edgelist', nodetype=int, create_using=nx.Graph())
    for e in G.edges:
        G.edges[e]['weight'] = 1 

    return G

def node_labels():
    '''Read node classes'''
    
    G = build_graph()
    d = {}
    with open('input/coralabels.edgelist', 'r') as lab:    
        l = lab.readlines()     
    
        if len(l[0].split(' ')) == 2:
                x = [int(w.split(' ')[0]) for w in l]
                y = [int(w.split(' ')[-1].rstrip('\n')) for w in l]  
                for j in range(len(x)):
                    d[x[j]] = y[j]      #For label data with node          
        elif len(l[0].split(' ')) == 1: 
            for i, k in enumerate(l, 1):    #start counting from 1
                d[i] = int(k.rstrip('\n'))    #For label data without node
        else: print('Edges more than 2 nodes')
        labels_dict = {}
        for node in G.nodes:
            labels_dict[node] = d[node]
        return labels_dict
            
        
def node_neighbor_labels():
    G = build_graph()
    labels_dict = node_labels()
    neighbor_labels = {}
    for node in G.nodes:
        nn = list(G.neighbors(node))    
        nnlabel = [labels_dict[n] for n in nn]  #Superimpose labels on node id
        neighbor_labels[node] = nnlabel
    return neighbor_labels, labels_dict