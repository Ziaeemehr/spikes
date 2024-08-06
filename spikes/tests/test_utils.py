import random
import unittest
import networkx as nx
from spikes.utils import *


# class TestFindSAP(unittest.TestCase):
#     def test_empty_graph(self):
#         G = nx.Graph()
#         start_node = "A"
#         target_node = "B"
#         all_saps = list(find_sap(G, start_node, target_node))
#         self.assertEqual(all_saps, [])

#     def test_directly_connected_nodes(self):
#         G = nx.Graph()
#         edges = [("A", "B"), ("A", "C"), ("B", "D"), ("B", "E"), ("C", "F"), ("E", "F")]
#         G.add_edges_from(edges)
#         start_node = "A"
#         target_node = "B"
#         all_saps = list(find_sap(G, start_node, target_node))
#         self.assertEqual(len(all_saps), 2)
#         self.assertEqual(all_saps[0], ["A", "B"])
