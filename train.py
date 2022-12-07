import argparse
from gensim.models import Word2Vec 
from attribwalk import *
from builder import *
import networkx as nx

'''Enter input network and features in builder 
To preserve neighborhood topology, walk type == structure
To preserve node contexts, walk type == attribute 
To preserve both neighborhood topology and contextual attributes, walk type == hybrid'''


def parse_args():
    '''
    Parses arguments.
    '''
    parser = argparse.ArgumentParser(description="Run SNEFAN.")
    
    parser.add_argument('--output', nargs='?', default='output/cora.emb',
                        help='Embedding path')            #default='emb/karate.emb'

    parser.add_argument('--dimensions', type=int, default=64,
                        help='Number of dimensions. Default is 64.')

    parser.add_argument('--walk-length', type=int, default=40,
                        help='Length of walk per source. Default is 40.')

    parser.add_argument('--num-walks', type=int, default=5,
                        help='Number of walks per source. Default is 5.')

    parser.add_argument('--window-size', type=int, default=5,
                        help='Context size for optimization. Default is 5.')

    parser.add_argument('--epochs', default=1, type=int,
                      help='Number of epochs in SGD')
    
    parser.add_argument("--walk-type", nargs = "?", default = "hybrid",
                    help = "Random walk order... choose either structure or attribute or hybrid")

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--min-count', type=int, default=0,
                        help='Minimum count of Training words. Default is 0.')
    
    parser.add_argument('--sg', type=int, default=1,
                        help='Training Algorithm. CBOW=0,SkipGram=1. Default is 1.')
    
    return parser.parse_args()


def learn_embeddings(walks):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    walks = [list(map(str, walk)) for walk in walks]
    print("Training Node Corpus...")
    model = Word2Vec(walks, vector_size=args.dimensions, window=args.window_size, 
                     min_count=args.min_count, sg=args.sg, workers=args.workers, epochs=args.epochs,  
                    sample=1e-5, alpha=0.25, min_alpha=0.01, negative=5)
    print("Saving Embeddings...")
    model.wv.save_word2vec_format(args.output)
    
    return model


def main(args):
    G = build_graph()
    walks = ATTRIB_NEIGH(args.num_walks, args.walk_length, args.walk_type)
    learn_embeddings(walks)


if __name__ == "__main__":
    args = parse_args()
    main(args)
    


