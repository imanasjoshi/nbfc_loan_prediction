import streamlit as st
import pandas as pd

st.set_page_config(page_title="Portfolio Explorer", layout="wide")
st.title("Portfolio Explorer")

@st.cache_data
def load_data():
    return pd.read_csv("data/raw/Train.csv")

try:
    df = load_data()
    st.write("Browse the raw dataset used for modeling.")
    
    col1, col2 = st.columns(2)
    with col1:
        grade = st.multiselect("Filter by Grade", df['loan_grade'].dropna().unique())
    with col2:
        default_status = st.multiselect("Filter by Default", df['default'].unique())
        
    filtered = df.copy()
    if grade:
        filtered = filtered[filtered['loan_grade'].isin(grade)]
    if default_status:
        filtered = filtered[filtered['default'].isin(default_status)]
        
    st.dataframe(filtered.head(100))
    st.caption("Showing up to 100 records based on filters.")
    
except Exception as e:
    st.error("Error loading data.")
