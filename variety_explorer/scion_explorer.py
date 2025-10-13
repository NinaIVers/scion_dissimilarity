import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import matplotlib.pyplot as plt

# Performance: cache data loading
@st.cache_data
def load_data():
    return pd.read_csv('variety_explorer/df_scion_normalized.csv', encoding="ISO-8859-1", delimiter=";")

df = load_data()

# Page config
st.set_page_config(page_title="Grapevine Scion Explorer", layout="wide")

# Title and intro
st.title("🍇 Grapevine Scion Variety Explorer")
st.markdown("""
[![Source Code](https://img.shields.io/badge/source_code-mediumpurple?style=for-the-badge&logo=GitHub&logoColor=black&labelColor=lightsteelblue)](https://github.com/NinaIVers/scion_dissimilarity.git)

Explore genetic dissimilarity among 64 grapevine scion varieties using interactive visualizations and clustering filters.
""")

# Sidebar filters
st.sidebar.header("🔎 Filter Options")

# Initialize session state
if "selected_varieties" not in st.session_state:
    st.session_state.selected_varieties = df['Prime name'].unique().tolist()
if "selected_kmeans_group" not in st.session_state:
    st.session_state.selected_kmeans_group = 'All'
if "selected_ward_group" not in st.session_state:
    st.session_state.selected_ward_group = 'All'

# Clear filters button
if st.sidebar.button("🧹 Clear Filters"):
    st.session_state.selected_varieties = df['Prime name'].unique().tolist()
    st.session_state.selected_kmeans_group = 'All'
    st.session_state.selected_ward_group = 'All'

# Render widgets with current session state
selected_varieties = st.sidebar.multiselect(
    "Select Scion Varieties:",
    options=df['Prime name'].unique(),
    default=st.session_state.selected_varieties
)

selected_kmeans_group = st.sidebar.selectbox(
    "Select K-means Cluster:",
    options=['All'] + sorted(df['Kmeans cluster'].unique()),
    index=(['All'] + sorted(df['Kmeans cluster'].unique())).index(st.session_state.selected_kmeans_group)
)

selected_ward_group = st.sidebar.selectbox(
    "Select Ward Cluster:",
    options=['All'] + sorted(df['Ward cluster'].unique()),
    index=(['All'] + sorted(df['Ward cluster'].unique())).index(st.session_state.selected_ward_group)
)

# Update session state with current selections
st.session_state.selected_varieties = selected_varieties
st.session_state.selected_kmeans_group = selected_kmeans_group
st.session_state.selected_ward_group = selected_ward_group

# Apply filters
filtered_df = df.copy()
if selected_varieties:
    filtered_df = filtered_df[filtered_df['Prime name'].isin(selected_varieties)]
if selected_kmeans_group != 'All':
    filtered_df = filtered_df[filtered_df['Kmeans cluster'] == selected_kmeans_group]
if selected_ward_group != 'All':
    filtered_df = filtered_df[filtered_df['Ward cluster'] == selected_ward_group]


# Numeric columns
excluded_columns = ['Ward cluster', 'Kmeans cluster']
numeric_columns = [col for col in filtered_df.select_dtypes(include='number').columns if col not in excluded_columns]

# Tabs for layout
tab1, tab2, tab3, tab4 = st.tabs(["📋 Summary", "📈 Interactive Charts", "📊 Distributions", "🧬 Information"])

# Tab 1: Summary
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Filtered Scion Variety Statistics")
        st.dataframe(filtered_df.describe(include=['int64', 'float64']).round(5))
        
    with col2:
        st.subheader("📈 Scatter Plot")
        selected_var = st.selectbox("Select a feature to visualize:", numeric_columns)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(filtered_df.index, filtered_df[selected_var], alpha=0.5)
        ax.set_title(f'{selected_var} Scatter Plot', fontsize=16)
        ax.set_xlabel('Index', fontsize=12)
        ax.set_ylabel(selected_var, fontsize=12)
    
        mean_value = filtered_df[selected_var].mean()
        ax.axhline(y=mean_value, color='navy', linestyle='--', label=f'Average = {mean_value:.2f}')
        ax.legend()
        st.pyplot(fig)


# Tab 2: Interactive Charts
with tab2:

    selected_y = st.selectbox("Select feature to visualize:", numeric_columns)
    st.markdown(f"#### 📊 Distribution of {selected_y} by K-means Cluster")
    fig_box = px.box(filtered_df, x='Kmeans cluster', y=selected_y,
                     color='Kmeans cluster', points='all')
    st.plotly_chart(fig_box)

    st.markdown(f"#### 📊 Distribution of {selected_y} by Ward Cluster")
    fig_box = px.box(filtered_df, x='Ward cluster', y=selected_y,
                     color='Ward cluster', points='all')
    st.plotly_chart(fig_box)

# Tab 3: Distributions
with tab3:
    
    selected_feature = st.selectbox("Select feature to visualize:", numeric_columns, key="hist_var")
    col1, col2 = st.columns(2)

    with col1:
        
        st.markdown(f"#### 📊 Histogram of {selected_feature}")
        fig, ax = plt.subplots(figsize=(5, 3))
        filtered_df[selected_feature].plot(kind='hist',
                                        orientation='horizontal',
                                        color='mediumpurple',
                                        edgecolor='black',
                                        density=True,
                                        histtype='bar',
                                        stacked=True,
                                        ax=ax)
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Value')
        plt.grid(True)
        st.pyplot(fig, use_container_width=True)

    with col2:
        st.markdown(f"#### 📊 Boxplot of {selected_feature}")
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        filtered_df[[selected_feature]].boxplot(
                                                fontsize=10,
                                                notch=True,
                                                capprops=dict(linewidth=0.5),
                                                meanprops=dict(marker='D', markerfacecolor='darkorchid', markersize=5.8),
                                                grid=True,
                                                medianprops=dict(linestyle='-', linewidth=3.5, color='orange'),
                                                flierprops=dict(marker='o', markerfacecolor='orchid', markersize=7, markeredgecolor='darkorchid'),
                                                boxprops=dict(color='black'),
                                                ax=ax2
                                            )
        plt.xticks(fontsize=10)
        plt.yticks(rotation=45, fontsize=10)
        st.pyplot(fig2, use_container_width=True)


with tab4:
    
    
# Footer
st.markdown("""
---
**Note:** This tool is part of a research project on genetic dissimilarity of grapevine scion varieties using unsupervised machine learning. 🔬🍇 """)
