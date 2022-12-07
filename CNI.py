'''Compact neighborhood Encoding using Bijection Function'''

import math
import numpy as np

from builder import *
neighbor_labels, labels_dict = node_neighbor_labels()
only_labels = list(set(labels_dict.values()))
G = build_graph()

    
def cantor_pairing(x, y):
    if x > (y + x - 1):
        return 0
    if y + x - 1 == 1:
        return 1
    st = 1
    i = 1
    t = y
    while t <= (y + x - 1):
        st *= t
        t += 1       
        while (i <= x) and (st % i == 0) and (st > 0):
            st = st / i
            i += 1
    return st


def CNR():
    CNI_Nodes_log = {}
    CNI_original = {}
    for node in G.nodes:
        visited = []
        CNI_list = []
        CNI_Sum = 0
        hop_labels = sorted(neighbor_labels[node])  #Returns Labels of Node neighbors
        hop_count = np.zeros(len(only_labels))
        for label in only_labels:
            if label in hop_labels:
                hop_count[only_labels.index(label)] = hop_labels.count(label)   #returns counts of unique neighbor labels 
            else: 0
        hop_count = [int(n) for n in hop_count]    
        for i in range(len(only_labels)):
            visited.append(hop_count.pop(0))
            CNI_list.append((only_labels[i],sum(visited)))    #Returns the label, hop count sum pair
        for j in CNI_list:
            CNI_Sum += cantor_pairing(j[0], j[1])
        
        CNI_original[node] = int(CNI_Sum)
        if CNI_Sum == 1:
            CNI_Nodes_log[node] = CNI_Sum
        else:
            CNI_Nodes_log[node] = math.log(CNI_Sum)
    return (CNI_original, CNI_Nodes_log)


