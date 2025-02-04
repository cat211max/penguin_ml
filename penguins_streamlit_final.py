import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
st.title('Penguin Classifier')
st.write('This app uses the Palmer Penguins dataset to classify penguin species based on user inputs')

penguin_df =pd.read_csv("penguins.csv")

rf_pickle = open('random_forest_penguin.pickle','rb')

map_pickle = open('output_penguin.pickle','rb')
rfc = pickle.load(rf_pickle)

unique_penguin_mapping = pickle.load(map_pickle)

rf_pickle.close()
map_pickle.close()
island = st.selectbox("Penguin Island",options = ['Biscoe','Dream','Torgersen'])
sex = st.selectbox("Sex", options=["Female","Male"])
bill_length = st.number_input("Bill Length (mm)",min_value=0.0)
bill_depth = st.number_input("Bill Depth (mm)",min_value=0.0)
flipper_length = st.number_input("Flipper Length (mm)",min_value=0.0)
body_mass = st.number_input("Body Mass (g)",min_value=0.0)
user_inputs = [island,sex,bill_length,bill_depth,flipper_length,body_mass]
st.write(f"""the user inputs are {user_inputs}""".format())  

# get the inputs into correct format 
island_biscoe, island_dream, island_torgersen = 0,0,0   
if island == 'Biscoe':
    island_biscoe = 1
elif island == 'Dream':
    island_dream = 1
elif island == 'Torgersen':
    island_torgersen = 1
sex_female, sex_male = 0,0
if sex == 'Female':
    sex_female = 1
elif sex == 'Male':
    sex_male = 1
    
# prediction 
new_prediction = rfc.predict([[bill_length,bill_depth,flipper_length,body_mass,island_biscoe,island_dream,island_torgersen,
                               sex_female,sex_male]])

prediction_species = unique_penguin_mapping[new_prediction][0]
st.write(f"The predicted species is {prediction_species}")