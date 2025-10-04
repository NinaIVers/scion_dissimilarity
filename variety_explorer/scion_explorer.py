# -*- coding: utf-8 -*-
"""scion_explorer.ipynb
"""

# Commented out IPython magic to ensure Python compatibility.
#pip install streamlit
#pip install plotly

import altair as alt
import streamlit as st
import pandas as pd
import plotly.express as px

# Load dataset
df = pd.read_csv('variety_explorer/df_scion_normalized.csv',
                 encoding= "ISO-8859-1", delimiter=";")

# Page configuration
st.set_page_config(layout = "wide")


st.title("Grapevine Scion Variety Explorer 🍇🔍")
st.markdown("""
        [![Source Code](https://img.shields.io/badge/source_code-mediumpurple?style=for-the-badge&logo=GitHub&logoColor=black&labelColor=lightsteelblue)](https://github.com/NinaIVers/scion_dissimilarity.git)
        """)

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


# Filters
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

def create_point_chart(data, x, y):
    chart = alt.Chart(data).mark_circle(size=100).encode(
        x=alt.X(x, title=x),
        y=alt.Y(y, title=y),
        color=alt.Color('Ward cluster:N', title='Ward Cluster'),
        tooltip=['Prime name', 'Kmeans cluster', 'Ward cluster']
    ).properties(
        width=800,
        height=500
    ).interactive()
    return chart

st.altair_chart(create_point_chart(filtered_df, x="Kmeans cluster", y="Prime name"))

#Boxplot
import plotly.express as px

fig = px.box(filtered_df, x='Kmeans cluster', y='End of maturation',
             color='Kmeans cluster', points='all',
             title='Distribution of Maturation by K-means Cluster')
st.plotly_chart(fig)

# Heatmap



# Footer
st.markdown("""
---
**Note:** This tool is part of a research project on genetic dissimilarity of grapevine scion varieties using unsupervised machine learning. 
""")
