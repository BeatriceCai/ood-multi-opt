from rdkit import Chem
from ligand_graph_features import *
from torch_geometric.loader import DataLoader
import warnings
warnings.filterwarnings('ignore')
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def featurizer(smiles):
    chem_graph_list = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if type(mol) == type(None):
            print('no mol')
        else:
            graph = mol_to_graph_data_obj_simple(mol)
            chem_graph_list.append(graph)
    chem_graphs_loader = DataLoader(chem_graph_list, batch_size=len(smiles),
                                    shuffle=False)
    for batch in chem_graphs_loader:
        chem_graphs = batch

    return chem_graphs