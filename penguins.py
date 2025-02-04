
import streamlit as st

import pandas as pd

st.title("Palmer's Penguins")   

# import our data
penguins_df = pd.read_csv('penguins.csv')

st.write(penguins_df.head())

