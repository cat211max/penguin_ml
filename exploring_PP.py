import streamlit as st
import pandas as pd
import altair as alt
import seaborn as sns
import time 

st.title("Palmer's Penguins")

st.markdown("This is a dataset of the Palmer's Penguins collected from Palmer Station in Antarctica. The dataset contains information about the species, island, bill length, bill depth, flipper length, body")
 
 # penguins_df = pd.read_csv('penguins.csv')
penguin_file = st.file_uploader("Select Your local penguins csv(default provided)")

@st.cache_data(penguin_file)
def load_file(penguin_file):
    time.sleep(3)
    if penguin_file is not None:
        df = pd.read_csv(penguin_file)
    else:
        df = pd.read_csv('penguins.csv') 
    return(df)      

penguins_df = load_file(penguin_file) 


select_x_var = st.selectbox('What do you want the x-axis to be?', ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g'])

select_y_var = st.selectbox('What do you want the y-axis to be?', ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g'])

# filter out penguins based on their gender
select_gender = st.selectbox('What  gender do you want to filter for?', ['all penguins', 'male penguins', 'female penguins'])

if select_gender == 'male penguins':
    penguins_df = penguins_df[penguins_df['sex'] == 'male']
elif select_gender == 'female penguins':
    penguins_df = penguins_df[penguins_df['sex'] == 'female']
else:
    pass

    


alt_chart = (
    alt.Chart(penguins_df, title=f"Scatterplot of Palmer's Penguins")
    .mark_circle()
    .encode(
        x=select_x_var,
        y=select_y_var, 
        color ="species"
    )
    .interactive()

)

st.altair_chart(alt_chart, use_container_width=True)
