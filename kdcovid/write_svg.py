

def format_node(node_id, url, bipartite, color, label, shape):
    return """%s [URL="%s", bipartite=%s, color=%s, label=%s, rank=sink, shape=%s, style=filled];\n""" % (node_id, url, bipartite, color, label, shape)

def format_gene(node_id, url, label):
    return format_node(node_id, url, 1, 'lightsalmon', label, 'rectangle')

def format_drug(node_id, url, label):
    return format_node(node_id, url, 0, 'palegreen', label, 'egg')

def format_disease(node_id, url, label):
    return format_node(node_id, url, 2, '#ffffcc', label, 'oval')

def format_real_edge(node1, node2, url, label):
    return """%s -- %s  [URL="%s", color=black, label=%s];\n""" % (node1, node2, url, label)

def format_invis_edge(node1, node2):
    return """%s -- %s  [color=white, constraint=True, label="", style=invis];\n""" % (node1, node2)

def format_graph(gene_nodes, drug_nodes, disease_nodes, real_edges, invis_edges):
    s = """strict graph  {\ndpi=64;\nrankdir=LR;\n"""
    for node_id, url, label in gene_nodes:
        s += format_gene(node_id, url, label)
    for node_id, url, label in drug_nodes:
        s += format_drug(node_id, url, label)
    for node_id, url, label in disease_nodes:
        s += format_disease(node_id, url, label)
    for node1, node2, url, label in real_edges:
        s += format_real_edge(node1, node2, url, label)
    for node1, node2 in invis_edges:
        s += format_invis_edge(node1, node2)
    s += "}"
    return s
