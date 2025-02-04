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
