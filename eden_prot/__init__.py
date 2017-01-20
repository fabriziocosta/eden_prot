#!/usr/bin/env python
"""Provides encoding into graphs for proteins."""

from Bio.PDB import Selection, NeighborSearch
from collections import defaultdict
from collections import deque
import networkx as nx
from networkx import union
from toolz import curry, compose

# ac properties
aa_properties_dict = {
    "fullname": {"G": "glycine", "A": "alanine",
                 "V": "valine", "I": "isoleucine",
                 "L": "leucine", "M": "methionine",
                 "F": "phenylalanine", "Y": "tyrosine",
                 "W": "tryptophan", "S": "serine",
                 "T": "threonine", "N": "asparagine",
                 "Q": "glutamine", "K": "lysine",
                 "R": "arginine", "H": "histidine",
                 "D": "aspartic acid", "E": "glutamic acid",
                 "C": "cytosine", "P": "proline"},
    "oneLetterCode": {"Gly": "G", "Ala": "A", "Val": "V",
                      "Ile": "I", "Leu": "L", "Met": "M",
                      "Phe": "F", "Tyr": "Y", "Trp": "W",
                      "Ser": "S", "Thr": "T", "Asn": "N",
                      "Gln": "Q", "Lys": "K", "Arg": "R",
                      "His": "H", "Asp": "D", "Glu": "E",
                      "Cys": "C", "Pro": "P"},
    "threeLetterCode": {"G": "Gly", "A": "Ala", "V": "Val",
                        "I": "Ile", "L": "Leu", "M": "Met",
                        "F": "Phe", "Y": "Tyr", "W": "Trp",
                        "S": "Ser", "T": "Thr", "N": "Asn",
                        "Q": "Gln", "K": "Lys", "R": "Arg",
                        "H": "His", "D": "Asp", "E": "Glu",
                        "C": "Cys", "P": "Pro"},
    "residue_class": {"G": "aliphatic", "A": "aliphatic",
                      "V": "aliphatic", "I": "aliphatic",
                      "L": "aliphatic", "M": "thioether",
                      "F": "aromatic", "Y": "aromatic",
                      "W": "aromatic", "S": "alcohol",
                      "T": "alcohol", "N": "amide", "Q": "amide",
                      "K": "amine", "R": "amine", "H": "amine",
                      "D": "carboxylic acid", "E": "carbxylic acid",
                      "C": "thiol", "P": "cyclic aliphatic"},
    "polarity": {"G": 0, "A": 0, "V": 0, "I": 0, "L": 0, "M": 0,
                 "F": 0, "Y": 1, "W": 0, "S": 1, "T": 1, "N": 1,
                 "Q": 1, "K": 1, "R": 1, "H": 1, "D": 1, "E": 1,
                 "C": 1, "P": 0},
    "charge": {"G": 0, "A": 0, "V": 0, "I": 0, "L": 0, "M": 0, "F": 0,
               "Y": 0, "W": 0, "S": 0, "T": 0, "N": 1, "Q": 0, "K": 1,
               "R": 1, "H": 1, "D": -1, "E": -1, "C": 0, "P": 0},
    # hydrophobicity after Hoop and Woods
    "hydrophobicity_Hoop": {"G": 0, "A": -0.5, "V": -1.5, "I": -1.8,
                            "L": -1.8, "M": -1.3, "F": -2.5, "Y": -2.3,
                            "W": -3.4, "S": 0.3, "T": -0.4, "N": 0.2,
                            "Q": 0.2, "K": 3.0, "R": 3.0, "H": -0.5,
                            "D": 3.0, "E": 3.0, "C": -1.0, "P": 0.0},
    # hydrophobicity after Kyle and Doolittle
    "hydrophobicity_Kyle": {"G": -0.4, "A": 1.8, "V": 4.2, "I": 4.5,
                            "L": 3.8, "M": 4.5, "F": 2.8, "Y": -1.3,
                            "W": -0.9, "S": -0.8, "T": -0.7, "N": -3.5,
                            "Q": -3.5, "K": -3.9, "R": -4.5, "H": -3.2,
                            "D": -3.5, "E": -3.5, "C": 2.5, "P": -1.6},
    # hydrophobicity after Engelmann
    "hydrophobicity_Engelmann": {"G": 1, "A": 1.6, "V": 2.6, "I": 3.1,
                                 "L": 2.8, "M": 3.4, "F": 3.7, "Y": -0.7,
                                 "W": 1.9, "S": 0.6, "T": 1.2, "N": -4.8,
                                 "Q": -4.1, "K": -8.8, "R": -12.3,
                                 "H": -3.0, "D": -9.2, "E": -8.2,
                                 "C": 2.0, "P": -0.2},
    # residue surface in square AAngstroem
    "surface_area": {"G": 75.0, "A": 115.0, "V": 155.0, "I": 175.0,
                     "L": 170.0, "M": 185.0, "F": 210.0, "Y": 230.0,
                     "W": 255.0, "S": 115.0, "T": 140.0, "N": 150.0,
                     "Q": 190.0, "K": 200.0, "R": 225.0, "H": 195.0,
                     "D": 160, "E": 180, "C": 135.0, "P": 145.0},
    # residue volume in cubic AAngstroem
    "volume": {"G": 60.1, "A": 88.6, "V": 140.0, "I": 166.7, "L": 166.7,
               "M": 162.9, "F": 189.9, "Y": 193.3, "W": 227.8, "S": 89.0,
               "T": 116.1, "N": 111.1, "Q": 138.4, "K": 168.6, "R": 173.4,
               "H": 153.2, "D": 114.1, "E": 143.8, "C": 108.5, "P": 112.7}
}

ac_dict = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
           'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
           'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
           'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}


def ac_encoding(code, scheme='code_20'):
    """ac_encoding."""
    # codes from L. R. Murphy, A. Wallqvist, and R. M. Levy, Protein Eng. 2000.
    code_3 = ['LASGVTIPMC', 'EKRDNQH', 'FYW']
    code_5 = ['LVIMC', 'ASGTP', 'FYW', 'EDNQ', 'KRH']
    code_6 = ['LVIM', 'ASGT', 'PHC', 'FYW', 'EDNQ', 'KR']
    code_12 = ['LVIM', 'C', 'A', 'G', 'ST', 'P', 'FY', 'W',
               'EQ', 'DN', 'KR', 'H']

    if ac_dict.get(code, False) is False:
        return code
    ac_1letter_code = ac_dict[code]
    if scheme == 'code_20':
        return ac_1letter_code
    elif scheme == 'code_3':
        for n, codes in enumerate(code_3):
            if ac_1letter_code in codes:
                return n
    elif scheme == 'code_5':
        for n, codes in enumerate(code_5):
            if ac_1letter_code in codes:
                return n
    elif scheme == 'code_6':
        for n, codes in enumerate(code_6):
            if ac_1letter_code in codes:
                return n
    elif scheme == 'code_12':
        for n, codes in enumerate(code_12):
            if ac_1letter_code in codes:
                return n
    else:
        raise Exception('Unknown scheme: %s' % scheme)


# ligand
def _get_atom_id(atom, ligand_id):
    id = atom.get_id() + '_' + str(ligand_id)
    return str(id)


def _get_atom_label(atom):
    label = atom.element
    return str(label)


def make_ligand_graph(structure=None, atoms=None, ligand_id=None):
    """Make the molecular graph from a PDB structure and a list of atoms."""
    # constants
    _triple_bond_threshold = 1.18
    _double_bond_threshold = 1.32
    _single_bond_threshold = 1.8

    graph = nx.Graph()
    # add nodes
    for atom in atoms:
        i = _get_atom_id(atom, ligand_id)
        label = _get_atom_label(atom)
        graph.add_node(i,
                       label=label,
                       chain_id=ligand_id,
                       atom=atom,
                       typeof='ligand',
                       coords=atom.get_coord())
    # add edges
    for n, atom1 in enumerate(atoms):
        i = _get_atom_id(atom1, ligand_id)
        for atom2 in atoms[n + 1:]:
            j = _get_atom_id(atom2, ligand_id)
            atom_distance = atom2 - atom1
            bond_type = None
            if(atom_distance < _triple_bond_threshold):
                bond_type = '#'
            elif(atom_distance < _double_bond_threshold):
                bond_type = '='
            elif(atom_distance < _single_bond_threshold):
                bond_type = '-'
            else:
                bond_type = None
            if bond_type:
                graph.add_edge(i, j, label=bond_type, typeof='ligand')
    return graph


def _extract_atoms(graph):
    atom_dict = defaultdict(list)
    for u in graph.nodes():
        chain_id = graph.node[u]['chain_id']
        atom = graph.node[u]['atom']
        atom_dict[chain_id].append(atom)
    atom_list = list()
    for chain_id in atom_dict:
        atom_list.append(atom_dict[chain_id])
    return atom_list


def _extract_ligand_atoms(structure, ligand_marker="PXG"):
    for model in structure:
        for chain_id, chain in enumerate(model):
            for res in chain:
                if res.get_resname() == ligand_marker:
                    atoms = Selection.unfold_entities(res, 'A')
                    yield atoms


def make_ligand_graphs(structure, ligand_marker="PXG"):
    """make_ligand_graphs.

    If there are multiple chains that are marked with the ligand_marker,
    extract all of those as individual graphs.
    """
    atoms_list = _extract_ligand_atoms(structure, ligand_marker=ligand_marker)
    atoms_list = list(atoms_list)
    for ligand_chain_id, atoms in enumerate(atoms_list):
        graph = make_ligand_graph(structure, atoms, ligand_chain_id)
        yield graph


def make_ligands_graph(structure, ligand_marker="PXG"):
    """make_ligands_graph.

    Make the union of all ligand graphs that have the same ligand_marker.
    """
    def union_redux(graphs):
        return reduce(union, graphs, graphs.next())

    graphs = make_ligand_graphs(structure, ligand_marker)
    return union_redux(graphs)

# protein


def _get_residue_id(residue, chain_id):
    res_tuple = residue.get_full_id()
    res_name = res_tuple[3]
    res_id = res_name[1]
    res_id = str(res_id) + '_' + str(chain_id)
    return res_id


def _extract_residues(structure):
    res_seq = []
    for model in structure:
        for chain_id, chain in enumerate(model):
            for res in chain:
                atoms = [atom for atom in res if atom.get_name() == 'CA']
                if atoms:
                    ca_atom = atoms[0]
                    res_id = _get_residue_id(res, chain_id)
                    res_seq.append((res_id,
                                    res.get_resname(),
                                    ca_atom,
                                    chain_id))
    return res_seq


def _is_valid_residue(residue):
    res_tuple = residue.get_full_id()
    res_name = res_tuple[3]
    if ' ' in res_name[0]:
        return True
    else:
        return False


def _add_edges(orig_graph, min_dist=1, max_dist=7, typeof='conjunctive'):
    graph = orig_graph.copy()
    node_list = graph.nodes()
    for i in range(len(node_list)):
        u = node_list[i]
        for j in range(i + 1, len(node_list)):
            v = node_list[j]
            dist = graph.node[u]['atom'] - graph.node[v]['atom']
            if min_dist <= dist < max_dist:
                if typeof == 'conjunctive':
                    graph.add_edge(u, v,
                                   label='-',
                                   len=dist,
                                   typeof='protein_conj')
                elif typeof == 'disjunctive':
                    graph.add_edge(u, v,
                                   label='-',
                                   len=dist,
                                   nesting=True,
                                   typeof='protein_disj')
    return graph


def _add_nodes(orig_graph, structure):
    graph = orig_graph.copy()
    res_seq = _extract_residues(structure)
    for res_id, ac, atom, chain_id in res_seq:
        ac1code = ac_dict[ac]
        vec = [aa_properties_dict['charge'][ac1code],
               aa_properties_dict['polarity'][ac1code],
               aa_properties_dict['hydrophobicity_Hoop'][ac1code],
               aa_properties_dict['hydrophobicity_Kyle'][ac1code],
               aa_properties_dict['hydrophobicity_Engelmann'][ac1code],
               aa_properties_dict['surface_area'][ac1code],
               aa_properties_dict['volume'][ac1code]]
        graph.add_node(
            res_id,
            label=ac,
            vec=vec,
            atom=atom,
            ac=ac,
            typeof='residue',
            name=ac1code,
            chain_id=chain_id,
            charge=aa_properties_dict['charge'][ac1code],
            polarity=aa_properties_dict['polarity'][ac1code],
            hydr_Hoop=aa_properties_dict['hydrophobicity_Hoop'][ac1code],
            hydr_Kyle=aa_properties_dict['hydrophobicity_Kyle'][ac1code],
            hydr_Engel=aa_properties_dict['hydrophobicity_Engelmann'][ac1code],
            surface_area=aa_properties_dict['surface_area'][ac1code],
            volume=aa_properties_dict['volume'][ac1code],
            contact=False,
            coords=atom.get_coord())
    return graph


def make_protein_graph(structure,
                       min_dist_conj=2,
                       max_dist_conj=7,
                       min_dist_disj=None,
                       max_dist_disj=8):
    """make_protein_graph."""
    if min_dist_disj is None:
        min_dist_disj = max_dist_conj
    graph = nx.Graph()
    graph = _add_nodes(graph, structure)
    graph = _add_edges(graph, min_dist_conj, max_dist_conj,
                       typeof='conjunctive')
    graph = _add_edges(graph, min_dist_disj, max_dist_disj,
                       typeof='disjunctive')
    return graph


def relabel_protein_graph(orig_graph, scheme='code_20'):
    """relabel_protein_graph."""
    graph = orig_graph.copy()
    for u in graph.nodes():
        ac = graph.node[u]['ac']
        label = ac_encoding(ac, scheme=scheme)
        graph.node[u]['label'] = label
    return graph

# ligand protein interaction


def join_ligand_protein(structure,
                        orig_protein_graph,
                        orig_ligand_graph,
                        interaction_distance_th=5.4):
    """join_ligand_protein."""
    # do the union of ligand_graph to protein_graph
    graph = nx.union(orig_protein_graph, orig_ligand_graph)
    atoms_list = _extract_atoms(orig_ligand_graph)
    for ligand_chain_id, atoms in enumerate(atoms_list):
        models = structure.get_list()
        # iterate over the models
        for model in models:
            # get all the chains in this model
            chains = model.get_list()
            # get all the atoms in this model
            model_atoms = Selection.unfold_entities(model, 'A')
            # create a NeighborSearch
            ns = NeighborSearch(model_atoms)
            # search the chains in for the ligand_id
            for chain_id, aChain in enumerate(chains):
                if ligand_chain_id == chain_id:
                    for atom in atoms:
                        atom_id = _get_atom_id(atom, chain_id)
                        neighbors = ns.search(atom.get_coord(),
                                              interaction_distance_th)
                        residue_list = Selection.unfold_entities(
                            neighbors, 'R')
                        for aResidue in residue_list:
                            if _is_valid_residue(aResidue):
                                res_id = _get_residue_id(aResidue, chain_id)
                                graph.add_edge(atom_id, res_id,
                                               label=':',
                                               nesting=True,
                                               typeof='ligand_protein')
                                graph.node[res_id]['contact'] = True
    return graph


def _mark_single_vertex_breadth_first_visit(graph,
                                            root=None,
                                            attribute=None,
                                            max_depth=None,
                                            key_nesting='nesting'):
    graph.node[root][attribute] = True
    visited = set()  # use a set as we can end up exploring few nodes
    # q is the queue containing the frontieer to be expanded in the BFV
    q = deque()
    q.append(root)
    # the map associates to each vertex id the distance from the root
    dist = {}
    dist[root] = 0
    visited.add(root)
    while len(q) > 0:
        # extract the current vertex
        u = q.popleft()
        d = dist[u] + 1
        if d <= max_depth:
            # iterate over the neighbors of the current vertex
            for v in graph.neighbors(u):
                if v not in visited:
                    # skip nesting edge-nodes
                    if graph.edge[u][v].get(key_nesting, False) is False:
                        dist[v] = d
                        visited.add(v)
                        graph.node[v][attribute] = True
                        q.append(v)


def _mark_active(graph,
                 max_depth=None,
                 root_attribute='contact',
                 root_value=True,
                 attribute='active',
                 key_nesting='nesting'):
    # mark all nodes as False
    for u in graph.nodes():
        graph.node[u][attribute] = False
    # mark as True all nodes that are within distance 'max_depth'
    # from a node that has root_attribute=root_value
    for u in graph.nodes():
        node_dict = graph.node[u]
        if root_attribute in node_dict and \
                node_dict[root_attribute] == root_value:
            _mark_single_vertex_breadth_first_visit(graph,
                                                    root=u,
                                                    attribute=attribute,
                                                    max_depth=max_depth)
    return graph


def _extract_subgraph(original_graph):
    selected_nodes = list()
    for u in original_graph.nodes():
        if original_graph.node[u]['active'] is False and \
                original_graph.node[u]['typeof'] == 'residue':
            pass
        else:
            selected_nodes.append(u)
    graph = original_graph.subgraph(selected_nodes)
    return graph


def trim_ligand_protein_graph(ligand_protein_graph, depth=0):
    """trim_ligand_protein_graph."""
    cc_mark_active = curry(_mark_active)(max_depth=depth)
    trim = compose(_extract_subgraph, cc_mark_active)
    graph = trim(ligand_protein_graph)
    return graph


def make_trimmed_ligand_protein_graph(structure,
                                      ligand_marker,
                                      min_dist_conj=4,
                                      max_dist_conj=6.3,
                                      min_dist_disj=None,
                                      max_dist_disj=8,
                                      depth=0,
                                      interaction_distance_th=5.4):
    """make_trimmed_ligand_protein_graph."""
    ligand_graph = make_ligands_graph(structure, ligand_marker)
    protein_graph = make_protein_graph(structure,
                                       min_dist_conj,
                                       max_dist_conj,
                                       min_dist_disj,
                                       max_dist_disj)
    protein_graph = relabel_protein_graph(protein_graph, scheme='code_5')
    ligand_protein_graph = join_ligand_protein(structure,
                                               protein_graph,
                                               ligand_graph,
                                               interaction_distance_th)
    graph = trim_ligand_protein_graph(ligand_protein_graph, depth=depth)
    return graph
