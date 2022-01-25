import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import GCNConv, Sequential as GraphSequential

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG

import matplotlib
import matplotlib.cm as cm

np.random.seed(1234)

# Feauterizers from course "Machine Learning in Drug Design" at Jagiellonian University, Kraków, Poland.
class Featurizer:
    def __init__(self, y_column, **kwargs):
        self.y_column = y_column
        self.__dict__.update(kwargs)
    
    def __call__(self, df):
        raise NotImplementedError()

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise ValueError("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))
        

class GraphFeaturizer(Featurizer):
    def __call__(self, smiles):
        graphs = []
        mol = Chem.MolFromSmiles(smiles)

        edges = []
        for bond in mol.GetBonds():
            edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            edges.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
        edges = np.array(edges)

        nodes = []
        for atom in mol.GetAtoms():
            results = one_of_k_encoding_unk(
                atom.GetSymbol(),
                [
                    'Br', 'C', 'Cl', 'F', 'H', 'I', 'N', 'O', 'P', 'S', 'Unknown'
                ]
            ) + one_of_k_encoding(
                atom.GetDegree(),
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            ) + one_of_k_encoding_unk(
                atom.GetImplicitValence(),
                [0, 1, 2, 3, 4, 5, 6]
            ) + [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + one_of_k_encoding_unk(
                atom.GetHybridization(),
                [
                    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                    Chem.rdchem.HybridizationType.SP3D2
                ]
            ) + [atom.GetIsAromatic()] + one_of_k_encoding_unk(
                atom.GetTotalNumHs(),
                [0, 1, 2, 3, 4]
            )
            nodes.append(results)

        nodes = np.array(nodes)
            
        graphs.append((nodes, edges.T))
        return graphs

class GraphNeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(GraphNeuralNetwork, self).__init__()
        torch.manual_seed(42)
        self.final_conv_acts = None
        self.final_conv_grads = None

        self.conv1 = GCNConv(42, 100)
        self.conv2 = GCNConv(100, 400)
        self.conv3 = GCNConv(400, 200)
        self.conv4 = GCNConv(200, 100)

        self.out = nn.Linear(100, 50)
        self.out2 = nn.Linear(50, 1)
    
    def activations_hook(self, grad):
        self.final_conv_grads = grad

    def forward(self, x, edge_index, batch_index):
        out = self.conv1(x, edge_index)
        out = F.relu(out)
        out = self.conv2(out, edge_index)
        out = F.relu(out)
        out = self.conv3(out, edge_index)
        out = F.relu(out)
        out = F.dropout(out, training=self.training) #p=0.5
        
        with torch.enable_grad():
            self.final_conv_acts = self.conv4(out, edge_index)
        self.final_conv_acts.register_hook(self.activations_hook)
        
        out = F.relu(self.final_conv_acts)
        out = gap(out, batch_index) # global_mean_pool
        out = self.out(out)
        out = F.relu(out)
        out = self.out2(out)

        return out


def grad_cam(final_conv_acts, final_conv_grads):
    node_heat_map = []
    alphas = torch.mean(final_conv_grads, axis=0)
    for n in range(final_conv_acts.shape[0]):
        node_heat = F.relu(alphas @ final_conv_acts[n]).item()
        node_heat_map.append(node_heat)
    return node_heat_map


def draw_gradcam(atom_weights, mol):
    atom_weights = np.array(atom_weights)
    if (atom_weights > 0.).any():
        atom_weights = atom_weights / atom_weights.max() / 2

    if len(atom_weights) > 0:
        norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
        cmap = cm.get_cmap('bwr')
        plt_colors = cm.ScalarMappable(norm=norm, cmap=cmap)
        atom_colors = {
            i: plt_colors.to_rgba(atom_weights[i]) for i in range(len(atom_weights))
        }
        highlight_kwargs = {
            'highlightAtoms': list(range(len(atom_weights))),
            'highlightBonds': [],
            'highlightAtomColors': atom_colors
        }

    d = rdMolDraw2D.MolDraw2DSVG(500, 500) 
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol, **highlight_kwargs)
    d.FinishDrawing()
    svg = d.GetDrawingText()
    svg = svg.replace('svg:', '')
    return svg


def predict_and_interpret(smiles):
	model = GraphNeuralNetwork()
	model.load_state_dict(torch.load('graph_model_128.pt'))
	model.eval()

	featurizer = GraphFeaturizer(y_column='pIC50')
	graph = featurizer(smiles)
	X, E = graph[0]
	data = Data(x=torch.FloatTensor(X), edge_index=torch.LongTensor(E))

	x, edge_index, batch = data.x, data.edge_index, data.batch

	pred = model(x, edge_index, torch.zeros(x.shape[0], dtype=torch.int64))
	pred.backward()
	prediction = pred.data.cpu().numpy()[0]

	atom_weights = grad_cam(model.final_conv_acts, model.final_conv_grads)    
	    
	return prediction, atom_weights

def predict(data_frame):
	model = GraphNeuralNetwork()
	model.load_state_dict(torch.load('graph_model_128.pt'))
	model.eval()

	graph_predictions = []

	for smiles in data_frame['smiles']:
		featurizer = GraphFeaturizer(y_column='pIC50')
		graph = featurizer(smiles)
		X, E = graph[0]
		data = Data(x=torch.FloatTensor(X), edge_index=torch.LongTensor(E))
		x, edge_index, batch = data.x, data.edge_index, data.batch
		pred = model(x, edge_index, torch.zeros(x.shape[0], dtype=torch.int64))
		pred.backward()
		graph_predictions.append(pred.data.cpu().numpy()[0][0])

	return np.array(graph_predictions) 