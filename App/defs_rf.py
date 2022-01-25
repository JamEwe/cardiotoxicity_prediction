
import pandas as pd

import numpy as np
import pickle
import dill

import lime
from lime import lime_tabular

import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem


Maccs_keys_dict = {
          1:('?','Isotope'),
          2:('[#104]','Atomic no 104 - Rutherfordium'),
          3:('[#32,#33,#34,#50,#51,#52,#82,#83,#84]','Group IVa, Va, VIa Periods 4-6: Ge, As, Se, Sn, Sb, Te,Pb, Bi, Po'), 
          4:('[Ac,Th,Pa,U,Np,Pu,Am,Cm,Bk,Cf,Es,Fm,Md,No,Lr]','Actinide'),
          5:('[Sc,Ti,Y,Zr,Hf]','Group IIIB, IVB'),  
          6:('[La,Ce,Pr,Nd,Pm,Sm,Eu,Gd,Tb,Dy,Ho,Er,Tm,Yb,Lu]','Lanthanide'),
          7:('[V,Cr,Mn,Nb,Mo,Tc,Ta,W,Re]','Group VB, VIB, VIIB'),
          8:('[!#6;!#1]1~*~*~*~1','QAAA@1'),
          9:('[Fe,Co,Ni,Ru,Rh,Pd,Os,Ir,Pt]','Group VIII'),
          10:('[Be,Mg,Ca,Sr,Ba,Ra]','Group IIa (Alkaline earth)'),
          11:('*1~*~*~*~1','4M Ring'),
          12:('[Cu,Zn,Ag,Cd,Au,Hg]','Group IB, IIB'),
          13:('[#8]~[#7](~[#6])~[#6]','ON(C)C'), 
          14:('[#16]-[#16]','S-S'), 
          15:('[#8]~[#6](~[#8])~[#8]','OC(O)O'),
          16:('[!#6;!#1]1~*~*~1','QAA@1'),
          17:('[#6]#[#6]','CTC'),
          18:('[#5,#13,#31,#49,#81]','Group IIIA'),
          19:('*1~*~*~*~*~*~*~1','7M Ring'),
          20:('[#14]','Si'), 
          21:('[#6]=[#6](~[!#6;!#1])~[!#6;!#1]','C=C(Q)Q'),
          22:('*1~*~*~1','3M Ring'),
          23:('[#7]~[#6](~[#8])~[#8]','NC(O)O'),
          24:('[#7]-[#8]','N-O'), 
          25:('[#7]~[#6](~[#7])~[#7]','NC(N)N'),
          26:('[#6]=;@[#6](@*)@*','C$=C($A)$A'),
          27:('[I]','I'),
          28:('[!#6;!#1]~[CH2]~[!#6;!#1]','QCH2Q'),
          29:('[#15]','P'),
          30:('[#6]~[!#6;!#1](~[#6])(~[#6])~*','CQ(C)(C)A'),
          31:('[!#6;!#1]~[F,Cl,Br,I]','QX'),
          32:('[#6]~[#16]~[#7]','CSN'),
          33:('[#7]~[#16]','NS'),
          34:('[CH2]=*','CH2=A'), 
          35:('[Li,Na,K,Rb,Cs,Fr]','Group IA (Alkali Metal)'), 
          36:('[#16R]','S Heterocycle'), 
          37:('[#7]~[#6](~[#8])~[#7]','NC(O)N'),
          38:('[#7]~[#6](~[#6])~[#7]',' NC(C)N'),
          39:('[#8]~[#16](~[#8])~[#8]','OS(O)O'), 
          40:('[#16]-[#8]','S-O'), 
          41:('[#6]#[#7]','CTN'), 
          42:('F','F'), 
          43:('[!#6;!#1;!H0]~*~[!#6;!#1;!H0]','QHAQH'), 
          44:('?','Other - atoms include any atoms other than H, C, N, O, Si, P, S, F, Cl, Br, and I, and is abbreviated "Z"'),
          45:('[#6]=[#6]~[#7]','C=CN'), 
          46:('Br',"Br"), 
          47:('[#16]~*~[#7]','SAN'), 
          48:('[#8]~[!#6;!#1](~[#8])(~[#8])','OQ(O)O'),
          49:('[!+0]','CHARGE'), 
          50:('[#6]=[#6](~[#6])~[#6]','C=C(C)C'),
          51:('[#6]~[#16]~[#8]','CSO'),
          52:('[#7]~[#7]','NN'),
          53:('[!#6;!#1;!H0]~*~*~*~[!#6;!#1;!H0]','QHAAAQH'),
          54:('[!#6;!#1;!H0]~*~*~[!#6;!#1;!H0]','QHAAQH'), 
          55:('[#8]~[#16]~[#8]','OSO'), 
          56:('[#8]~[#7](~[#8])~[#6]','ON(O)C'), 
          57:('[#8R]','O Heterocycle'), 
          58:('[!#6;!#1]~[#16]~[!#6;!#1]','QSQ'), 
          59:('[#16]!:*:*','Snot%A%A'), 
          60:('[#16]=[#8]','S=O'), 
          61:('*~[#16](~*)~*','AS(A)A'), 
          62:('*@*!@*@*','A$!A$A'), 
          63:('[#7]=[#8]','N=O'), 
          64:('*@*!@[#16]','A$A!S'), 
          65:('c:n','C%N'), 
          66:('[#6]~[#6](~[#6])(~[#6])~*','CC(C)(C)A'),
          67:('[!#6;!#1]~[#16]','QS'), 
          68:('[!#6;!#1;!H0]~[!#6;!#1;!H0]','QHQH (&...)'),
          69:('[!#6;!#1]~[!#6;!#1;!H0]','QQH'),
          70:('[!#6;!#1]~[#7]~[!#6;!#1]','QNQ'),
          71:('[#7]~[#8]','NO'),
          72:('[#8]~*~*~[#8]','OAAO'),
          73:('[#16]=*','S=A'), 
          74:('[CH3]~*~[CH3]','CH3ACH3'), 
          75:('*!@[#7]@*','A!N$A'),
          76:('[#6]=[#6](~*)~*','C=C(A)A'), 
          77:('[#7]~*~[#7]','NAN'), 
          78:('[#6]=[#7]', 'C=N'), 
          79:('[#7]~*~*~[#7]', 'NAAN'),
          80:('[#7]~*~*~*~[#7]','NAAAN'), 
          81:('[#16]~*(~*)~*', 'SA(A)A'), 
          82:('*~[CH2]~[!#6;!#1;!H0]','ACH2QH'),
          83:('[!#6;!#1]1~*~*~*~*~1','QAAAA@1'),
          84:('[NH2]','NH2'),
          85:('[#6]~[#7](~[#6])~[#6]','CN(C)C'), 
          86:('[C;H2,H3][!#6;!#1][C;H2,H3]','CH2QCH2'), 
          87:('[F,Cl,Br,I]!@*@*','X!A$A'),
          88:('[#16]','S'),
          89:('[#8]~*~*~*~[#8]','OAAAO'), 
          90:('[$([!#6;!#1;!H0]~*~*~[CH2]~*),$([!#6;!#1;!H0;R]1@[R]@[R]@[CH2;R]1),$([!#6;!#1;!H0]~[R]1@[R]@[CH2;R]1)]','QHAACH2A'), 
          91:('[$([!#6;!#1;!H0]~*~*~*~[CH2]~*),$([!#6;!#1;!H0;R]1@[R]@[R]@[R]@[CH2;R]1),$([!#6;!#1;!H0]~[R]1@[R]@[R]@[CH2;R]1),$([!#6;!#1;!H0]~*~[R]1@[R]@[CH2;R]1)]','QHAAACH2A'),
          92:('[#8]~[#6](~[#7])~[#6]','OC(N)C'),
          93:('[!#6;!#1]~[CH3]', 'QCH3'), 
          94:('[!#6;!#1]~[#7]','QN'),
          95:('[#7]~*~*~[#8]', 'NAAO'), 
          96:('*1~*~*~*~*~1', '5M Ring'), 
          97:('[#7]~*~*~*~[#8]', 'NAAAO'), 
          98:('[!#6;!#1]1~*~*~*~*~*~1', 'QAAAAA@1'), 
          99:('[#6]=[#6]', 'C=C'), 
          100:('*~[CH2]~[#7]', 'ACH2N'), 
          101:('[$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1)]','8M Ring or larger (max 14)'),
          102:('[!#6;!#1]~[#8]', 'QO'), 
          103:('Cl', 'Cl'), 
          104:('[!#6;!#1;!H0]~*~[CH2]~*', 'QHACH2A'), 
          105:('*@*(@*)@*', 'A$A($A)$A'), 
          106:('[!#6;!#1]~*(~[!#6;!#1])~[!#6;!#1]', 'QA(Q)Q'), 
          107:('[F,Cl,Br,I]~*(~*)~*', 'XA(A)A'), 
          108:('[CH3]~*~*~*~[CH2]~*','CH3AAACH2A'), 
          109:('*~[CH2]~[#8]', 'ACH2O'), 
          110:('[#7]~[#6]~[#8]', 'NCO'), 
          111:('[#7]~*~[CH2]~*', 'NACH2A'), 
          112:('*~*(~*)(~*)~*', 'AA(A)(A)A'), 
          113:('[#8]!:*:*', 'Onot%A%A'), 
          114:('[CH3]~[CH2]~*', 'CH3CH2A'), 
          115:('[CH3]~*~[CH2]~*', 'CH3ACH2A'), 
          116:('[$([CH3]~*~*~[CH2]~*),$([CH3]~*1~*~[CH2]1)]','CH3AACH2A'), 
          117:('[#7]~*~[#8]','NAO'),
          118:('[$(*~[CH2]~[CH2]~*),$(*1~[CH2]~[CH2]1)]','ACH2CH2A'),
          119:('[#7]=*','N=A'), 
          120:('[!#6;R]','Heterocyclic atom > 1 (&...)'),
          121:('[#7;R]','N Heterocycle'), 
          122:('*~[#7](~*)~*', 'AN(A)A'), 
          123:('[#8]~[#6]~[#8]', 'OCO'), 
          124:('[!#6;!#1]~[!#6;!#1]', 'QQ'), 
          125:('?', 'Aromatic Ring > 1'), 
          126:('*!@[#8]!@*', 'A!O!A'), 
          127:('*@*!@[#8]', 'A$A!O > 1 (&...)'), 
          128:('[$(*~[CH2]~*~*~*~[CH2]~*),$([R]1@[CH2;R]@[R]@[R]@[R]@[CH2;R]1),$(*~[CH2]~[R]1@[R]@[R]@[CH2;R]1),$(*~[CH2]~*~[R]1@[R]@[CH2;R]1)]', 'ACH2AAACH2A'), 
          129:('[$(*~[CH2]~*~*~[CH2]~*),$([R]1@[CH2]@[R]@[R]@[CH2;R]1),$(*~[CH2]~[R]1@[R]@[CH2;R]1)]', 'ACH2AACH2A'), 
          130:('[!#6;!#1]~[!#6;!#1]', 'QQ > 1 (&...) '), 
          131:('[!#6;!#1;!H0]', 'QH > 1'), 
          132:('[#8]~*~[CH2]~*', 'OACH2A'), 
          133:('*@*!@[#7]', 'A$A!N'), 
          134:('[F,Cl,Br,I]', 'X (HALOGEN)'), 
          135:('[#7]!:*:*', 'Nnot%A%A'), 
          136:('[#8]=*', 'O=A>1'),  
          137:('[!C;!c;R]', 'Heterocycle'), 
          138:('[!#6;!#1]~[CH2]~*', 'QCH2A>1 (&...)'),
          139:('[O;!H0]', 'OH'), 
          140:('[#8]', 'O > 3 (&...)'),
          141:('[CH3]', 'CH3 > 2  (&...)'), 
          142:('[#7]', 'N > 1'),
          143:('*@*!@[#8]', 'A$A!O'), 
          144:('*!:*:*!:*', 'Anot%A%Anot%A'), 
          145:('*1~*~*~*~*~*~1', '6M ring > 1'),
          146:('[#8]', 'O > 2'), 
          147:('[$(*~[CH2]~[CH2]~*),$([R]1@[CH2;R]@[CH2;R]1)]', 'ACH2CH2A'), 
          148:('*~[!#6;!#1](~*)~*', 'AQ(A)A'), 
          149:('[C;H3,H4]', 'CH3 > 1'), 
          150:('*!@*@*!@*', 'A!A$A!A'), 
          151:('[#7;!H0]', 'NH'), 
          152:('[#8]~[#6](~[#6])~[#6]', 'OC(C)C'), 
          153:('[!#6;!#1]~[CH2]~*', 'QCH2A'), 
          154:('[#6]=[#8]', 'C=O'), 
          155:('*!@[CH2]!@*', 'A!CH2!A'), 
          156:('[#7]~*(~*)~*', 'NA(A)A'), 
          157:('[#6]-[#8]', 'C-O'), 
          158:('[#6]-[#7]', 'C-N'), 
          159:('[#8]', 'O>1'), 
          160:('[C;H3,H4]', 'CH3'), 
          161:('[#7]', 'N'), 
          162:('a', 'Aromatic'), 
          163:('*1~*~*~*~*~*~1', '6M Ring'), 
          164:('[#8]', 'O'), 
          165:('[R]', 'Ring'),
          166:('?', 'Fragments')
  }

info = """ Atom symbols:
A : Any valid periodic table element symbol 
Q : Hetro atoms; any non-C or non-H atom 
X : Halogens; F, Cl, Br, I 
Z : Others; other than H, C, N, O, Si, P, S, F, Cl, Br, I 

Bond types:

- : Single 
= : Double 
T : Triple 
# : Triple 
~ : Single or double query bond 
% : An aromatic query bond 
None : Any bond type; no explicit bond specified 
$ : Ring bond; before a bond type specifies ring bond 
! : Chain or non-ring bond; before a bond type specifies chain bond 
@ : A ring linkage and the number following it specifies the atoms position in the line, thus @1 means linked back to the first atom in the list.

Aromatic: Kekule or Arom5 
Kekule: Bonds in 6-membered rings with alternate single/double bonds or perimeter bonds 
Arom5: Bonds in 5-membered rings with two double bonds and a hetro atom at the apex of the ring. """

def GetMACCS(smiles):
    """Function to convert smiles to numpy array of MACCS keys"""
    mol = Chem.MolFromSmiles(smiles)
    fp = MACCSkeys.GenMACCSKeys(mol)
    fp.ToBitString() 
    fp_list = list(fp)[1:]
    return np.array(fp_list)

def Explain(fingerprint, model):
  """Function to return model explanation for fingerprint"""
  with open('lime_explainer', 'rb') as f:
    explainer = dill.load(f)
  explanation = explainer.explain_instance(fingerprint, model.predict, num_features=20)
  return explanation

def ExpToDf(explanation, fingerprint):
    """Function to convert explanation from lime to pandas dataframe with extra informations"""
    df_exp = pd.DataFrame(explanation.as_map()[1], columns=['Key', 'Influence'])
    df_exp['Key']+=1
    df_exp['Value'] = [fingerprint[idx-1] for idx in df_exp['Key']]
    df_exp['Influence sign'] = np.where(df_exp['Influence'] < 0, "negative", np.where(df_exp['Influence'] >0, 'positive', 'neutral')) 
    df_exp = df_exp[['Key', 'Value', 'Influence', 'Influence sign']]
    df_exp['Smarts'] = df_exp.apply(lambda x: Maccs_keys_dict[x.Key][0], axis=1)
    df_exp['Smarts'] = df_exp['Smarts'].str.replace('$', '$\$$')
    df_exp['Description'] = df_exp.apply(lambda x: Maccs_keys_dict[x.Key][1], axis=1)
    df_exp['Description'] = df_exp['Description'].str.replace('$', '$\$$')
    df1 = df_exp.style.set_table_styles([dict(selector='th', props=[('text-align', 'center')])])
    df1.set_properties(**{'text-align': 'center'}).hide_index()   
    return df1

def predict_and_interpret(smiles):
  """Function to get prediction and interpretation dataframe"""
  loaded_model = pickle.load(open("rf.pkl", 'rb'))
  fingerprint = GetMACCS(smiles)
  prediction  = loaded_model.predict(fingerprint.reshape(1,-1))
  explanation  = Explain(fingerprint, loaded_model)
  data_frame = ExpToDf(explanation, fingerprint)  	    
  return prediction, data_frame

def predict(data_frame):
  """Function to get prediction for dataframe"""
  loaded_model = pickle.load(open("rf.pkl", 'rb'))
  fingerprints = [GetMACCS(smiles) for smiles in data_frame['smiles']]
  rf_predictions  = loaded_model.predict(fingerprints)
  return np.array(rf_predictions) 



