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
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('variety_explorer/df_scion_normalized.csv',
                 encoding= "ISO-8859-1", delimiter=";")

# Page configuration
st.set_page_config(layout = "wide")

st.title("Grapevine Scion Variety Explorer 🍇🔍")
st.markdown("""
        [![Source Code](https://img.shields.io/badge/source_code-mediumpurple?style=for-the-badge&logo=GitHub&logoColor=black&labelColor=lightsteelblue)](https://github.com/NinaIVers/scion_dissimilarity.git)
        """)

st.markdown(""" This interactive tool allows you to explore genetic dissimilarity among 64 grapevine scion varieties.
Use the filters on the sidebar to select specific cultivars or clustering groups.""")

# Sidebar filters
st.sidebar.header("🔎 Filter Options")
selected_varieties = st.sidebar.multiselect("Select Scion Varieties:",
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

# Allowed features
excluded_columns = ['Ward cluster', 'Kmeans cluster']
numeric_columns = [col for col in filtered_df.select_dtypes(include='number').columns
                   if col not in excluded_columns]

# Display filtered data
st.subheader("Filtered Scion Variety Data")
stats = numeric_columns.describe(include=['int64','float64']).round(5)
st.dataframe(stats)


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
selected_y = st.selectbox("Select feature:", numeric_columns)

fig_box = px.box(filtered_df, x='Kmeans cluster', y=selected_y,
                 color='Kmeans cluster', points='all',
                 title=f'Distribution of {selected_y} by K-means Cluster')
st.plotly_chart(fig_box)

# Hist
selected_variable = st.selectbox("Select a variable to view its histogram:", numeric_columns)

fig, ax = plt.subplots(figsize=(10, 4))
filtered_df[selected_variable].plot(kind='hist',
                           orientation='horizontal',
                           color='mediumpurple',
                           edgecolor='black',
                           density=True,
                           histtype='bar',
                           stacked=True,
                           ax=ax)

ax.set_title(f'Histogram of {selected_variable}', fontsize=14)
ax.set_xlabel('Frequency', fontsize=12)
ax.set_ylabel('Value', fontsize=12)
plt.grid(True)

st.subheader("📊 Histogram of Selected Descriptor")
st.pyplot(fig)

# Boxplot

box_property = dict(color='black')
flier_property = dict(marker='o', markerfacecolor='orchid',
                      markersize=7, markeredgecolor='darkorchid')
median_property = dict(linestyle='-', linewidth=3.5, color='orange')
mean_point_property = dict(marker='D', markerfacecolor='darkorchid',
                           markersize=5.8)

selected_variable = st.selectbox("Select a variable to view its boxplot:", numeric_columns)

fig, ax = plt.subplots(figsize=(10, 5))
filtered_df[[selected_variable]].boxplot(
    fontsize=13,
    notch=True,
    capprops=dict(linewidth=0.5),
    meanprops=mean_point_property,
    grid=True,
    medianprops=median_property,
    flierprops=flier_property,
    boxprops=box_property,
    ax=ax
)

ax.set_title(f'Boxplot of {selected_variable}', fontsize=14)
plt.xticks(rotation=90)
plt.yticks(rotation=45)

st.subheader(" Boxplot of Selected Descriptor")
st.pyplot(fig)

#Parallel cordinate plot
fig = px.parallel_coordinates(filtered_df,
    dimensions=['End of maturation', 'Species', 'Parent 1', 'Parent 2'],
    color='Kmeans cluster',
    title='Multivariate Comparison of Cultivars')
st.plotly_chart(fig)

# Heatmap



# Footer
st.markdown("""
---
**Note:** This tool is part of a research project on genetic dissimilarity of grapevine scion varieties using unsupervised machine learning. 
""")
