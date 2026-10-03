import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Executive Overview", layout="wide")
st.title("Executive Overview")

@st.cache_data
def load_data():
    return pd.read_csv("data/raw/Train.csv")

try:
    df = load_data()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Loans", f"{len(df):,}")
    col2.metric("Total Loan Amount", f"${df['loan_amnt'].sum():,.0f}")
    col3.metric("Average Loan Size", f"${df['loan_amnt'].mean():,.0f}")
    col4.metric("Observed Default Rate", f"{(df['default'].mean() * 100):.2f}%")
    
    st.markdown("---")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Default Distribution")
        fig1 = px.pie(df, names='default', title="Default Rate")
        st.plotly_chart(fig1, use_container_width=True)
        
    with col_b:
        st.subheader("Default Rate by Loan Grade")
        grade_defaults = df.groupby('loan_grade')['default'].mean().reset_index()
        fig2 = px.bar(grade_defaults, x='loan_grade', y='default', title="Default Rate by Grade")
        fig2.update_layout(yaxis_tickformat='.1%')
        st.plotly_chart(fig2, use_container_width=True)
        
except Exception as e:
    st.error(f"Error loading data: {e}")
