# import the necessary libraries

import numpy as np
import pandas as pd
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import mols2grid

#----------------------------------------------------
import pandas as pd
import numpy as np
#----------------------------------------------------
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
#from sklearn.ensemble import HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
#----------------------------------------------------
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sn
#-------------------------------
import streamlit as st
import streamlit.components.v1 as components
import pickle
from PIL import Image
import base64
import io
 
#--------- Import trained lgbm and hgbr regressors 

with open('model.pkl','rb') as f:
         model = pickle.load(f)
with open('scaler.pkl','rb') as f:
          scaler = pickle.load(f)
#--------- we need to use this 101 descriptors for prediction
descriptors = ['MaxEStateIndex', 'MinEStateIndex', 'MinAbsEStateIndex', 'MolWt', 'NumValenceElectrons', 'NumRadicalElectrons', 'FpDensityMorgan1', 'BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_MRHI', 'BCUT2D_MRLOW', 'BalabanJ', 'BertzCT', 'Chi0', 'Chi2v', 'Chi3v', 'HallKierAlpha', 'Ipc', 'Kappa1', 'Kappa2', 'Kappa3', 'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA7', 'SlogP_VSA8', 'TPSA', 'EState_VSA10', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'VSA_EState10', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'VSA_EState9', 'FractionCSP3', 'NHOHCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHeteroatoms', 'NumRotatableBonds', 'NumSaturatedCarbocycles', 'NumSaturatedRings', 'RingCount', 'MolLogP', 'fr_Al_OH', 'fr_Ar_N', 'fr_COO', 'fr_C_O_noCOO', 'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_alkyl_halide', 'fr_allylic_oxid', 'fr_amide', 'fr_aniline', 'fr_bicyclic', 'fr_ester', 'fr_ether', 'fr_halogen', 'fr_ketone', 'fr_ketone_Topliss', 'fr_methoxy', 'fr_para_hydroxylation']

#----------------------------------------------------------------------
st.set_page_config(page_title='HOMO-LUMO Energy gap Prediction App',layout='wide')
st.sidebar.markdown('<h2 style="color:white;background-image: linear-gradient(to right, #5977e3 , red);padding: 4%;border-radius:20px;text-align:center"> Use this Sidebar for HOMO-LUMO Energy Gap Prediction </h2>',unsafe_allow_html=True)


st.markdown('<h4 style="color:white;background-color:#5977e3;border-radius:20px;padding: 4%;text-align:center"> HOMO-LUMO Energy gap prediction Web App </h4>',unsafe_allow_html=True)

#---------- Display my linkedin page on the sidebar and main page
st.markdown("""[Gashaw M.Goshu](https://www.linkedin.com/in/gashaw-m-goshu/), Ph.D in Chemistry & M.S. in Data Science""")

#------------ Define HOMO-LUMO energy gap prediction and explain why it is important
st.markdown("""### HOMO-LUMO Energy Gap Prediction using RDKit Molecular Descriptors 
**HOMO** stands *for highest occupied molecular orbital* and **LUMO** stands for *lowest unoccupied molecular orbital*. HOMO-LUMO molecular orbitals are called frontier molecular orbitals. They are involved in chemical bond formation. Especially, pericyclic reactions such as cycloaddition, electrocyclic reactions, and sigmatropic rearrangement are explained using HOMO-LUMO molecular orbitals. In addition, in UV-Visible spectroscopy, the absorbance of organic molecules that have extended conjugated double bonds can be rationalized using the HOMO-LUMO energy gap of the molecules.""")

figure1 = Image.open('plainHOMO-LUMOEnegygap.jpg')
st.image(figure1, caption='Figure 1. HOMO-LUMO Energy gap of organic compounds')

 
st.markdown(""" Best results were obtained by taking the average of the two models (Light GBM Regressor(LGBMR) and Histogram-based Gradient Boosting Regressor(HGBR)). """)

# Import test data that contains predicted values to plot actual and predicted dataset
test = pd.read_csv('actual_vs_predict.csv')

# -------- Plot the figure of the test dataset on the webpage
plt.figure(figsize=(8, 6))
sn.regplot(x=test.Actual , y=test.Predicted,line_kws={"lw":2,'ls':'--','color':'black',"alpha":0.7})
plt.xlabel('Predicted Energy gap', color='blue')
plt.ylabel('Observed Energy gap', color ='blue')
plt.title("Test dataset", color='red')
plt.grid(alpha=0.2)

# --------R^2 (coefficient of determination) regression score function: 
R2 =r2_score(test.Actual, test.Predicted)
R2 = mpatches.Patch(label="R2={:04.2f}".format(R2))

#------- Model performance using MAE
MAE = mean_absolute_error(test.Actual, test.Predicted)

MAE = mpatches.Patch(label="MAE={:04.2f}".format(MAE))
plt.legend(handles=[R2, MAE])
plt.show()
st.pyplot(plt)

st.markdown('Figure 2. Average prediction of LGBM and HGBR regressors on the 30% or 862 test dataset and this prediction is slightly better than the [published paper](https://www.sciencedirect.com/science/article/pii/S2451929420300851). Note that in the published paper, the authors used multiple fingerprints, but here I used RDKit molecular features that can easily be calculated using a single function. It looks that the RDKit descriptors capture important features of molecules that can affect the HOMO-LUMO energy gap of organic compounds.')

# ============ User input
data = st.sidebar.text_input('Enter SMILE Strings in single or double quotation separated by comma:',"['CCCCO']" )
st.sidebar.markdown('''`or upload SMILE strings in CSV format, note that SMILE strings of the molecules should be in 'SMILES' column:`''')
multi_data = st.sidebar.file_uploader("=====================================")

st.sidebar.markdown("""**If you upload your CSV file, click the button below to get the HOMO-LUMO energy gap prediction in kcal/mol** """)
button_clicked = st.sidebar.button('Predict HOMO-LUMO Energy gap')
m = st.markdown("""
<style>
div.stButton > button:first-child {
    border: 1px solid #2e048a;
    border-radius:10px;
}
</style>""", unsafe_allow_html=True)

# Generate canonical SMILES
def canonical_smiles(smiles):
    mols = [Chem.MolFromSmiles(smi) for smi in smiles] 
    smiles = [Chem.MolToSmiles(mol) for mol in mols]
    return smiles 

# ================= Get the names of the 200 descriptors from RDKit
def calc_rdkit2d_descriptors(smiles):
    mols = [Chem.MolFromSmiles(i) for i in smiles]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
    desc_names = calc.GetDescriptorNames()
    # Append 200 molecular descriptors to each molecule in a list
    Mol_descriptors =[]
    for mol in mols:
        # Calculate all 200 descriptors for each molecule
        mol=Chem.AddHs(mol)
        descriptors = np.array(calc.CalcDescriptors(mol))
        Mol_descriptors.append(descriptors)
    return Mol_descriptors,desc_names  

#============ A function that can generate a csv file for output file to download
# Reference: https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806/2
#           https://github.com/dataprofessor/ml-auto-app/blob/main/app.py
def filedownload(data,file):
    df = data.to_csv(index=False)
    f= base64.b64encode(df.encode()).decode()
    link = f'<a href ="data:file/csv; base64,{f}" download={file}> Download {file} file</a>'
    return link

def predict_homo_lumo():
    # predicts homo-lumo gap based on conditions:
    if data!= "['CCCCO']":
        df = pd.DataFrame(eval(data), columns =['SMILES'])
        
        # Generate canonical SMILES
        df['SMILES'] = canonical_smiles(df['SMILES'].values)
        

        #========= function call to calculate 200 molecular descriptors using SMILES
        Mol_descriptors,desc_names = calc_rdkit2d_descriptors(df['SMILES'])

        #========= Put the 200 molecular descriptors in  dataframe or table
        Dataset_with_200_descriptors = pd.DataFrame(Mol_descriptors,columns=desc_names)

        #========= Use only the 101 descriptors listed above
        
        test_dataset_with_101_descriptors = Dataset_with_200_descriptors[descriptors] # these descriptors should be used for predictions


        #======== The data was standardized using standard scaler
        test_scaled = scaler.transform(test_dataset_with_101_descriptors)
    
        #---------------------------------------------------------------------

        #======== Prediction of HOMO-LUMO energy gap using lightgbm model
        lgbm_preds = model.predict(test_scaled)
        predicted_values = np.round(lgbm_preds,2)

        df1 = pd.DataFrame(columns=['SMILES','Predicted'])
        df1['SMILES'] =df['SMILES'].values
        df1['Predicted']= predicted_values
    
        # display the structures and energy gap on the sidebar
        st.sidebar.write(df1)
        # display data on the grid
        st.markdown('<div style="border: 2px solid #4908d4;border-radius:10px;border-radius:10px;color:purple;padding: 3%;text-align:center">  See below the structure and predicted HOMO-LUMO gap of compound/s.</i></div>',unsafe_allow_html=True)
   
        #======= Display output with structure in table form
        # reference:https://github.com/dataprofessor/drugdiscovery
        raw_html = mols2grid.display(df1,
                                set=["Name", "img"],
                                subset=['img', 'Predicted'],
                                # change the precision and format (or other transformations)
                                style = {"Predicted": lambda x: "color: red"},
                                transform={"Predicted": lambda x: round(x, 2)},
                                n_cols=5, n_rows=3,
                                tooltip = ['Predicted'],fixedBondLength=25, clearBackground=False)._repr_html_()
        components.html(raw_html, width=900, height=900, scrolling=False)

        #======= show CSV file attachment
        st.sidebar.markdown(filedownload(df1,"predicted_HOMO-LUMO energygap.csv"),unsafe_allow_html=True)

    #===== Use uploaded SMILES to calculate their logS values
    elif button_clicked:
        df2 = pd.read_csv(multi_data)
        df2['SMILES'] = canonical_smiles(df2['SMILES'].values)
        Mol_descriptors,desc_names = calc_rdkit2d_descriptors(df2['SMILES'])
        Dataset_with_200_descriptors = pd.DataFrame(Mol_descriptors,columns=desc_names)

        #========= Use only the 101 descriptors listed above 
        test_dataset_with_101_descriptors = Dataset_with_200_descriptors[descriptors] # these descriptors should be used for predictions    
            
        #======== The data was standardized using standard scaler
        test_scaled = scaler.transform(test_dataset_with_101_descriptors)

        #======== Prediction of toxicity using model1(LightGBM) and model2(XGBC)
        lgbm_preds = model.predict(test_scaled)
        prediction = np.round(lgbm_preds,2)

        df3 = pd.DataFrame(columns=['SMILES','Predicted'])
        df3['SMILES'] =df2['SMILES'].values
        df3['Predicted']= prediction

        st.sidebar.markdown('''## See your output in the following table:''')
        #======= Display output in table form
        st.sidebar.write(df3)

        #======= show CSV file attachment
        st.sidebar.markdown('''## See your output in the following table:''')
        st.sidebar.markdown(filedownload(df3,"predicted_energygap.csv"),unsafe_allow_html=True)
        st.markdown('<div style="border: 2px solid #4908d4;border-radius:10px;border-radius:10px;color:purple;padding: 3%;text-align:center">  See below the structure and predicted HOMO-LUMO gap of compounds.</i></div>',unsafe_allow_html=True)

        #======= Display output with structure in table form
        # reference:https://github.com/dataprofessor/drugdiscovery
        raw_html = mols2grid.display(df3,
                                subset=['img', 'Predicted'],
                                style = {"Predicted": lambda x: "color: red"},
                                # change the precision and format (or other transformations)
                                transform={"Predicted": lambda x: round(x, 2)},
                                n_cols=5, n_rows=3,
                                tooltip = ['Predicted'],fixedBondLength=25, clearBackground=False)._repr_html_()
        components.html(raw_html,width=900, height=900, scrolling=False)

    else:
        st.markdown('<div style="border: 2px solid #4908d4;border-radius:20px;padding: 3%;text-align:center"><h5> If you want to test this model,  please use the sidebar. If you have few molecules, you can directly put the SMILES in a single or double quotation separated by comma in the sidebar. If you have many molecules, you can put their SMILES strings in a "SMILES" column, upload them and click the button which says "Predict HOMO-LUMO Energygap" shown in the sidebar. </h5> <h5 style="color:white;background-color:red;border-radius:10px;padding: 3%;opacity: 0.7;">Please also note that predcition is more reliable if the compounds to be predicted have similar structures with the training data</h5></div>',unsafe_allow_html=True)

# call function to predict homo-lumo energy gap
predict_homo_lumo()