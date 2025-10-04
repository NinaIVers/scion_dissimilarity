# -*- coding: utf-8 -*-
"""scion_explorer.ipynb
"""

# Commented out IPython magic to ensure Python compatibility.
#pip install streamlit
#pip install plotly

import streamlit as st
import pandas as pd
import plotly.express as px

# Load dataset
df = pd.read_csv('variety_explorer/df_scion_normalized.csv',
                 encoding= "ISO-8859-1", delimiter=";")

# Page configuration
st.set_page_config(layout = "wide")


st.title("Grapevine Scion Variety Explorer 🍇🔍")
st.markdown(textwrap.dedent("""\
        [![Source Code](https://img.shields.io/badge/source_code-mediumpurple?style=for-the-badge&logo=GitHub&logoColor=black&labelColor=lightsteelblue)](https://github.com/NinaIVers/scion_dissimilarity.git)
    """))

st.markdown(""" This interactive tool allows you to explore genetic dissimilarity among grapevine scion varieties.
Use the filters on the sidebar to select specific cultivars or clustering groups.""")


# Sidebar filters
st.sidebar.header("🔎 Filter Options")
selected_varieties = st.sidebar.multiselect("Select varieties:",
                                            df['Prime name'].unique())
selected_kmeans_group = st.sidebar.selectbox("Select K-means heterotic group:",
                                             ['All'] + sorted(df['Kmeans cluster'].unique()))
selected_ward_group = st.sidebar.selectbox("Select Ward heterotic group:",
                                           ['All'] + sorted(df['Ward cluster'].unique()))


# Apply filters
filtered_df = df.copy()

if selected_varieties:
    filtered_df = filtered_df[filtered_df['Prime name'].isin(selected_varieties)]

if selected_kmeans_group != 'All':
    filtered_df = filtered_df[filtered_df['Kmeans cluster'] == selected_kmeans_group]

if selected_ward_group != 'All':
    filtered_df = filtered_df[filtered_df['Ward cluster'] == selected_ward_group]

# Display filtered data
st.subheader("Filtered Cultivar Data")
st.dataframe(filtered_df)

# Plots

# Heatmap



# Footer
st.markdown("""
---
**Note:** This tool is part of a research project on genetic dissimilarity of grapevine scion varieties using unsupervised machine learning. 
""")
