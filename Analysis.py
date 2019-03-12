#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:30:02 2019

Style reference: 
    CoQA: A Conversational Question Answering Challenge
        Figure 3: Distribution of trigram prefixes of questions in SQuAD and CoQA

@author: zjy
"""

import matplotlib.pyplot as plt
import numpy as np
import argparse
import logging


# In[]: argument
def parse_hparams():
    parser = argparse.ArgumentParser(description='hparams')
    parser.add_argument('-read', help='read from', default="example.txt")
    parser.add_argument('-ngram', help='ngram', type=int, default=3)
    parser.add_argument('-max_display_num', help='max number of ngrams to display', 
                        type=int, default=4)
    parser.add_argument('-min_count', help='min word occurence', 
                        type=int, default=1)
    parser.add_argument('-adjust_value', help='adjust node value for better visulization', 
                        type=int, default=1)
    parser.add_argument('-adjust_ratio', help='the total ratio taken up by child nodes', 
                        type=float, default=0.8)
    args = parser.parse_args()
    return args


# read file
class readlines_iterative(object):
    def __init__(self, corpora_file):
        self.corpora_file = corpora_file
        
    def __iter__(self):
        for line in open(self.corpora_file, 'r', encoding='utf-8'): 
            yield line.strip()


# In[]: save ngrams through Trie structure
class Trie:
    def __init__(self, args):
        self.root = ['', 1, 'white', []]
        self.cmap = plt.get_cmap("tab20") # there are 20 colors at maximum
        self.max_display_num = args.max_display_num
        self.min_word_count = args.min_count
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def insert(self, words:list) -> None: # Inserts a word into the trie
        """ Trie insertion """
        curNode = self.root
        for word in words:
            idx, already_contain = isInSubnodes(word, curNode)
            if not already_contain:
                curNode[-1].append([word, 1, [], []])
            else:
                curNode[-1][idx][1] += 1
            curNode = curNode[-1][idx]

    def prune(self, adjust_value=True, adjust_ratio=0.9):
        """ 
        prune based on max_display_num and min_count
        Args:
            adjust_value: if True, adjust node value for better visulization
            adjust_ratio: value within [0,1], the total ratio taken up by child nodes
        
        """
        self.logger.info('\tpruning')
        def recursive(node):
            if len(node[-1]) == 0:
                return
            
            subnodes = node[-1]
            subnodes = sorted(subnodes, key=lambda x: -x[1])[:self.max_display_num]
            subnodes = [n for n in subnodes if n[1] >= self.min_word_count]
            
            # re-assign subnode value for better visualization: 
            # subnode values sum up to adjust_ratio of father_value
            if adjust_value:
                father_value = node[1] * adjust_ratio
                subnode_values = [subnode[1] for subnode in subnodes]
                for subnode in subnodes:
                    subnode[1] = father_value * subnode[1]/sum(subnode_values)
            
            node[-1] = subnodes
            for subnode in node[-1]:
                recursive(subnode)
        
        recursive(self.root)
    
    def assign_node_color(self):
        """ assign node color """
        self.logger.info('\tcolorizing')
        def recursive(node, father_color):
            if not node:
                return
            
            node[2] = father_color
            for subNode in node[-1]:
                recursive(subNode, father_color)

        numNode_level_1 = len(self.root[-1])
        colors = self.cmap(range(numNode_level_1)).tolist()

        for idx, subnode_level_1 in enumerate(self.root[-1]):
            # assign father node color
            subnode_level_1[2] = colors[idx]
            # assign child node color
            recursive(subnode_level_1, colors[idx])

    def postprocess(self):
        self.logger.info('postprocessing')
        # assign root value
        root_value = 0
        for subnode in self.root[-1]:
            root_value += subnode[1]
        self.root[1] = root_value
        
        # prune and colorize
        self.prune()
        self.assign_node_color()
        

def isInSubnodes(word, node):
    for i,subnode in enumerate(node[-1]):
        if word == subnode[0]:
            return i, True
    return -1, False


def computeNgramTrie(args, sentences: list, n: int=3, deliminator=" ") -> list:
    all_ngrams = Trie(args)
    preprocess_func = lambda l: l.split(deliminator) if deliminator else l
    all_ngrams.logger.info("constructing ngram Trie")
    for sentence in sentences:
        sentence = preprocess_func(sentence)
        ngrams = [sentence[i:i+n] for i in range(len(sentence)-n+1)]
        for ngram in ngrams:
            all_ngrams.insert(ngram)
    
    all_ngrams.postprocess()
    return all_ngrams


# In[]: visualization
def sunburst(nodes, total=np.pi * 2, offset=0, level=0, max_level=3, ax=None):
    """ 
    This elegant recursive sunburst comes from: 
        https://stackoverflow.com/questions/12926779/how-to-make-a-sunburst-plot-in-r-or-python
    Thanks for the author.
    """
    ax = ax or plt.subplot(111, projection='polar')
    
    if level > max_level:
        return
    
    if level == 0:
        label, value, _, subnodes = nodes
        ax.bar([0], [0.5], [np.pi * 2], color='white')
        ax.text(0, 0, label, ha='center', va='center')
        sunburst(subnodes, total=value, ax=ax,
                 level=level + 1, max_level=max_level)
    elif nodes:        
        d = np.pi * 2 / total
        labels = []
        widths = []
        colors = []
        local_offset = offset
        
        for label, value, color, subnodes in nodes:
            labels.append(label)
            widths.append(value * d)
            colors.append(color)
            sunburst(subnodes, total=total, offset=local_offset,
                     level=level + 1, max_level=max_level, ax=ax)
            local_offset += value

        values = np.cumsum([offset * d] + widths[:-1])
        heights = [1] * len(nodes)
        bottoms = np.zeros(len(nodes)) + level - 0.5
        rects = ax.bar(values, heights, widths, bottoms, linewidth=1,
                       color=colors, edgecolor='white', align='edge')
        for rect, label in zip(rects, labels):
            x = rect.get_x() + rect.get_width() / 2
            y = rect.get_y() + rect.get_height() / 2
            rotation = (90 + (360 - np.degrees(x) % 180)) % 360
            ax.text(x, y, label, rotation=rotation, ha='center', va='center') 

    if level == 0:
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location('N')
        ax.set_axis_off()


if __name__ == '__main__':
    # args
    args = parse_hparams()
    
    # read file
    texts = readlines_iterative(args.read)

    # generate ngram
    ngramTrie = computeNgramTrie(args, texts, n=args.ngram, deliminator=" ")
    
    # plot sunburst
    plt.figure(figsize=(20,20)) # should be large for readability
    sunburst(ngramTrie.root, max_level=args.ngram+1)
    
    # save
    plt.savefig('sunburst.png', format='png', dpi=250)
