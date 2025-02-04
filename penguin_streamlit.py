import streamlit as st
import pickle

import seaborn as sns
import matplotlib.pyplot as plt
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
    
st.write(
    """
    We used a machine learning random forest model to predict the species, the features used in this prediction are 
    ranked by relative importance below.
    """
)

st.image("feature_important.png")

st.write(
    """
    below are the histograms for each continuous variable separated by penguin species. 
    the vertical line represents your inputted value. 
    """
)



fig, ax = plt.subplots()
ax = sns.displot(x=penguin_df['bill_length_mm'],hue=penguin_df['species'])

plt.axvline(bill_length) # reference lines 
plt.title("Bill length by species")
st.pyplot(ax)
fig, ax = plt.subplots()
ax = sns.displot(x=penguin_df["bill_depth_mm"],hue=penguin_df['species'])

plt.axvline(bill_depth)
plt.title('Bill depth by species')

st.pyplot(ax)
fig, ax = plt.subplots()
ax = sns.displot(x=penguin_df['flipper_length_mm'], 
                 hue= penguin_df['species'])

plt.axvline(flipper_length)

plt.title('Flipper length by species')

st.pyplot(ax)
