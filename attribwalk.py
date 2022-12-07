import numpy as np
import math
from numpy.random import seed
from numpy.random import randn
from numpy import argmin, argmax
from tqdm import tqdm
import time 
import networkx as nx

from builder import *
from CNI import CNR 

_, CNI_log = CNR()
G = build_graph()
        

def shortest_path(source, destination): 
    if nx.has_path(G, source, destination):
        path = nx.shortest_path(G, source, destination)
        path_length = nx.shortest_path_length(G, source, destination) 
    else:
        path = []
        path_length = len(path)
    return path, path_length


def node_neighbors():
    neighbors = {}
    for node in G.nodes:
        neighbors[node] = list(G.neighbors(node))
    return neighbors


def merge_CNI_Path(nn):
    CNI_node_neighbors = {}
    for n in nn:
        CNI_node_neighbors[n] = CNI_log[n]
        
    sim_CNI = {}
    for key, value in CNI_node_neighbors.items():
        if value not in sim_CNI:
            sim_CNI[value] = [key]
        else:
            sim_CNI[value].append(key)
    
    dupv = [k for k, _ in sim_CNI.items() if len(sim_CNI[k])>1]
    
    for d in dupv:
        val_d = sim_CNI[d]
        val_d_array = np.array(val_d)
        np.random.shuffle(val_d_array)
        random_path = np.random.choice(val_d_array) 
        sim_CNI[d] = [random_path]
    source_node_neigh = [val for key, value in sim_CNI.items() for val in value] 
    
    return source_node_neigh


def skip_visited(snn, visited):
    if len(snn) != 1:
        if len(visited) > 1:
            last_visit = visited[-2]
            if last_visit in snn:
                snn.remove(last_visit)
                skip_visited(snn, visited)
    return snn


def neighborhood_walk(node, walk_length):
    walk = [node]
    visited = [node] 
    while len(walk) < walk_length:
        current_node = walk[-1]
        nn = node_neighbors()[current_node]
        if len(nn) == 0:
            break
        if len(nn) == 1:
            walk.append(nn[0])
        else:            
            snn = skip_visited(nn, visited)
            neigh_prob, current_prob = norm_prob(snn, current_node)
            source = walk[0]
            for k, v in neigh_prob.items(): 
                penalize_const = 10
                _, path_length = shortest_path(source, k) 
                neigh_prob[k] = v * np.power(path_length, penalize_const)
            norm_neigh_prob, norm_current = alias_prob(neigh_prob, current_prob) 
            coeff = multi_reg(norm_neigh_prob, norm_current)
            if len(visited) == 1:
                next_node_index = argmax(coeff)
            else:
                next_node_index = argmin(coeff)
            next_node = snn[next_node_index] 
            walk.append(next_node)
            visited.append(next_node)
            
    return walk


def attribute_walk(node, walk_length):
    walk = [node]
    visited = [node]
    while len(walk) < walk_length:
        current_node = walk[-1]
        nn = node_neighbors()[current_node]
        if len(nn) == 0:
            break
        current_node_neigh = merge_CNI_Path(nn)
        current_node_neigh = skip_visited(current_node_neigh, visited)
        for n in current_node_neigh:
            if CNI_log[n] == CNI_log[current_node]:
                next_node = n
                walk.append(next_node)
            break
        neigh_prob, current_prob = norm_prob(current_node_neigh, current_node)
        norm_neigh_prob, norm_current = alias_prob(neigh_prob, current_prob)
        coeff = multi_reg(norm_neigh_prob, norm_current)
        next_node_index = argmax(coeff)
        next_node = current_node_neigh[next_node_index] 
        walk.append(next_node)
        visited.append(next_node)
        
    return walk
     
                         
def norm_prob(nn, current_node):
    neigh_prob = {}
    for n in nn:
        neigh_prob[n] = CNI_log[n]

    const_sum = np.sum([v for k, v in neigh_prob.items()])
    for k, v in neigh_prob.items():
        neigh_prob[k] = v / const_sum 
        
    current_prob = CNI_log[current_node] / const_sum
    
    return neigh_prob, current_prob


def alias_prob(neigh_dict, current_prob):
    k = len(neigh_dict)
    seed(15)
    gauss_var = randn(k)
    for key, value in neigh_dict.items():
        neigh_dict[key] = value * gauss_var
    
    norm_current = current_prob * gauss_var
    
    return neigh_dict, norm_current


def multi_reg(norm_neigh_prob, norm_current):
    y = norm_current
    X = []
    for _, v in norm_neigh_prob.items():
        X.append(v)
    X = np.transpose(X) # transpose so input vectors
    X = np.c_[X, np.ones(X.shape[0])]  # add bias term
    model_summary = np.linalg.lstsq(X, y, rcond=None)[0]    #With Intercept
    coeff = model_summary[:-1]
    
    return coeff
    

def ATTRIB_NEIGH(num_walk, walk_length, walk_type):

    print("STARTING RANDOM WALK... ")
    print("Number of Nodes:", len(G.nodes))
    print("Embedding Type: ", walk_type.upper(), "EMBEDDING")
    time.sleep(3)
    
    nodes = list(G.nodes)
    walk_corpus = []
    
    for cw in range(1, num_walk+1):
        print("\n")
        print("Current Walk: " + str(cw) + " of " + str(num_walk))
        for node in tqdm(nodes):
            if walk_type == "attribute": 
                node_walk = attribute_walk(node, walk_length)   
            elif walk_type == "structure":
                node_walk = neighborhood_walk(node, walk_length)
            else:
                node_walk = attribute_walk(node, walk_length) + neighborhood_walk(node, walk_length)

            walk_corpus.append(node_walk)
           
    return walk_corpus



