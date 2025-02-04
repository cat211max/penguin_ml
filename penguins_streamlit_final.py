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

penguin_file = st.file_uploader("Upload a file", type=["csv"])


if penguin_file is  None:
    rf_pickle = open('random_forest_penguin.pickle','rb')
    map_pickle = open('output_penguin.pickle','rb') 
    rfc = pickle.load(rf_pickle)
    unique_penguin_mapping = pickle.load(map_pickle)
    map_pickle.close()
    rf_pickle.close()
    penguin_df =  pd.read_csv("penguins.csv")
else:
    penguin_df = pd.read_csv('penguins.csv')
    penguin_df = penguin_df.dropna()
    output = penguin_df['species']
    features = penguin_df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g','sex']]
    features = pd.get_dummies(features)
    output, uniques = pd.factorize(output)
    x_train, x_test, y_train, y_test = train_test_split(features, output, test_size=0.8)
    rfc = RandomForestClassifier(random_state=15)
    rfc.fit(x_train.values, y_train)
    y_pred = rfc.predict(x_test.values)
    score = round(accuracy_score(y_pred,y_test),2 ) 

    st.write(
        f"""We trained a Random Forest Classifier on the Palmer Penguins dataset with an accuracy of {score}"""
    )
    
with st.form('user_inputs'):
    island = st.selectbox("Penguin Island",options = ['Biscoe','Dream','Torgersen'])
    sex = st.selectbox('Sex', options=['Female','Male'])
    bill_length = st.number_input('Bill Length (mm)',min_value=0.0)
    bill_depth = st.number_input('Bill Depth (mm)',min_value=0.0)
    flipper_length = st.number_input('Flipper Length (mm)',min_value=0.0)
    body_mass = st.number_input('Body Mass (g)',min_value=0.0)

    st.form_submit_button('Predict Penguin Species')
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
    
new_prediction =rfc.predict(
    [[bill_length,bill_depth,flipper_length,body_mass,island_biscoe,island_dream,island_torgersen,sex_female,sex_male,]]
)


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
