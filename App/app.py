
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from rdkit import Chem
import base64


import defs_rf
import defs_graph


st.set_page_config(
    page_title = 'Cardiotoxicity prediction app',
    page_icon = '❤️',
    layout = 'wide'
)

# Logo 
st.image('cartoon_logo.png', use_column_width=True)


# Title 
st.write("""

#### This app predicts the Cardiotoxicity (pIC50) values of molecules.


""")

# About
with st.expander("About"):
     st.write("""
         Drug candidates often cause a blockage of the potassium ion channel of the human ether-a-go-go-related gene (hERG). The blockage leads to long QT syndrome, which is a severe 
         life-threatening cardiac side effect. Predict drug-induced hERG-related cardiotoxicity could facilitate drug discovery by filtering out toxic drug candidates.

         Our machine learning model predicts cardiotoxicity of compound and shows what is contributing to it (explanation). It returns pIC50 value in moles. The result is calculated by a 
         Convolutional Graph Neural Network (based on molecular graph) and by a Random Forest Regressor (based on MACCs fingerprint) and then averaged (picture below).
         The input to the application can be a single SMILES or a CSV file containing a "smiles" column. The correctness of both a single SMILE and the SMILE list is checked. For a single SMILES, 
         the application also returns an explanation and classification (a molecule is considered toxic if pIC50 >= 5). For the CSV file it is possible to download the results in CSV format.





     """)
     st.image("model.png", width=None)


# Sidebar
st.sidebar.write("## Get prediction for single SMILES")
st.sidebar.text_input("", key="smiles", help="Enter single SMILES", max_chars=100)
col1, col2, col3 = st.sidebar.columns(3)
with col2:
    button1 = st.button('Predict')
st.sidebar.write("## Get predictions for more SMILES from CSV file")
st.sidebar.file_uploader("", key="file", help="Upload CSV file with 'smiles' column")
col4, col5, col6 = st.sidebar.columns([1,2,1])
with col5:
    button2 = st.button('Predict more')

def svg_to_html(svg):
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = r'<img src="data:image/svg+xml;base64,%s"/>' % b64
    return html

def file_download(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download</a>'
    return href


if button1:
	smiles = st.session_state.smiles
	mol = Chem.MolFromSmiles(smiles)
	if mol and smiles:

		with st.spinner("Calculating prediction and explanation..."):	
			#result from graph neural network
			graph_prediction, atom_weights = defs_graph.predict_and_interpret(smiles)
			html_svg = svg_to_html(defs_graph.draw_gradcam(atom_weights, mol))

			# result form random forest regressor
			rf_prediction, data_frame = defs_rf.predict_and_interpret(smiles)

			prediction = (graph_prediction + rf_prediction)/2
			prediction = prediction[0].round(3)

		st.header('Input SMILES')
		st.markdown(f"<h1 style='text-align: center; font-size : 25px; color: green;'>{smiles}</h1>", unsafe_allow_html=True)
		st.header('Predicted value')
		if(prediction>=5):
			st.markdown(f"<h1 style='text-align: center; font-size : 35px; color: red;'>{prediction}</h1>", unsafe_allow_html=True)
		else:
			st.markdown(f"<h1 style='text-align: center; font-size : 35px; color: green;'>{prediction}</h1>", unsafe_allow_html=True)
		st.markdown(f"<h1 style='text-align: center; font-size : 20px; color: black;'>{'red - toxic ; green - non-toxic'}</h1>", unsafe_allow_html=True)
		st.header('Explanation based on molecular graph')
		st.write(html_svg, unsafe_allow_html=True)
		st.header("Explanation based on molecules's fingerprint")
		st.write(data_frame)
		st.subheader("Description info")
		st.text(defs_rf.info)

	else:
		st.error('Invalid Smiles!')

elif button2:
	smiles_file = st.session_state.file
	if smiles_file:
		df_smiles = pd.read_csv(smiles_file)
		correct = 1
		for smiles in df_smiles['smiles']:
			mol = Chem.MolFromSmiles(smiles)
			if not(mol and smiles):
				correct = 0
				break
		if correct == 1:
			with st.spinner("Calculating predictions..."):		
				graph_predictions = defs_graph.predict(df_smiles)
				rf_predictions = defs_rf.predict(df_smiles)

				predictions = (graph_predictions + rf_predictions)/2
			predictions = pd.DataFrame(predictions, columns=['pIC50'])
			st.header('Input')
			st.write(df_smiles)
			st.header('Predicted values')
			st.write(predictions.style.format({"pIC50": "{:.3f}"}))
			st.markdown(file_download(predictions), unsafe_allow_html=True)
		else:
			st.error('Invalid Smiles in file!')
	else:
		st.error('Invalid file!')
  
else:
	st.info('Upload input data in the sidebar to start!')




