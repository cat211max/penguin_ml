import streamlit as st

st.title('My to do list')

if 'my_todo_list' not in st.session_state:
    st.session_state.my_todo_list = ["Learn Streamlit", "Build a Streamlit app", "Deploy the app"]



new_todo = st.text_input('Add a new task to your to do list')

if st.button('Add the new To-do item'):
    st.write('Adding new item to the list')
    st.session_state.my_todo_list.append(new_todo)
    
st.write('Updated to do list:', st.session_state.my_todo_list)
