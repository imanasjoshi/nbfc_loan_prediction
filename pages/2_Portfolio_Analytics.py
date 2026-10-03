import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Portfolio Analytics", layout="wide")
st.title("Portfolio Analytics")

@st.cache_data
def load_data():
    return pd.read_csv("data/raw/Train.csv")

try:
    df = load_data()
    
    st.sidebar.header("Filters")
    selected_grade = st.sidebar.multiselect("Loan Grade", options=sorted(df['loan_grade'].dropna().unique()), default=sorted(df['loan_grade'].dropna().unique()))
    
    filtered_df = df[df['loan_grade'].isin(selected_grade)]
    
    st.subheader(f"Portfolio Analytics for Selected Grades")
    st.write(f"Showing {len(filtered_df):,} loans.")
    
    fig = px.histogram(filtered_df, x="loan_amnt", color="default", nbins=50, title="Loan Amount Distribution by Default")
    st.plotly_chart(fig, use_container_width=True)
    
except Exception as e:
    st.error(f"Error loading data: {e}")
