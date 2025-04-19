import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO

# Page Configuration
st.set_page_config(
    page_title="Growth Mindset Data Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Theme CSS
st.markdown("""
<style>
    /* Main container */
    [data-testid="stAppViewContainer"] {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* Text elements */
    h1, h2, h3, h4, h5, h6, p, div, .stMarkdown {
        color: #ffffff !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1d24 !important;
    }
    
    /* Cards */
    .card {
        background-color: #1e2229 !important;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        margin-bottom: 20px;
        border: 1px solid #2a2f3b;
    }
    
    /* Input fields */
    .stTextInput, .stSelectbox, .stTextArea, .stNumberInput {
        background-color: #1e2229 !important;
        color: white !important;
        border: 1px solid #2a2f3b !important;
    }
    
    /* File uploader */
    .stFileUploader > div > div {
        background-color: #1e2229 !important;
        border: 2px dashed #4b6cb7 !important;
        color: white !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #4b6cb7 !important;
        color: white !important;
        border: none !important;
    }
    .stButton > button:hover {
        background-color: #3a5a9b !important;
    }
    
    /* Dataframes */
    .stDataFrame {
        background-color: #1e2229 !important;
        color: white !important;
    }
    
    /* Expanders */
    .stExpander {
        background-color: #1e2229 !important;
        border: 1px solid #2a2f3b !important;
    }
    
    /* Footer */
    .footer {
        background-color: #1a1d24 !important;
        color: white !important;
        padding: 15px;
        text-align: center;
        border-top: 1px solid #2a2f3b;
    }
    
    /* Plotly chart background */
    .js-plotly-plot .plotly {
        background-color: #1e2229 !important;
    }
    
    /* Matplotlib dark theme */
    .dark-plot {
        background-color: #1e2229 !important;
    }
    
    /* Tooltips */
    .stTooltip {
        background-color: #2a2f3b !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Apply dark theme to matplotlib
plt.style.use('dark_background')
sns.set_style("darkgrid")

# Sidebar
with st.sidebar:
    st.title("🧠 Growth Mindset Analyzer")
    st.markdown("**Develop your abilities through dedication and hard work!**")
    
    user_name = st.text_input("Your Name", placeholder="Enter your name", key="name_input")
    
    uploaded_file = st.file_uploader(
        "Upload Your Data (CSV/Excel)",
        type=["csv", "xlsx"],
        help="Maximum file size: 200MB",
        key="file_uploader"
    )

# Main Content
st.title("📊 Growth Mindset Data Analyzer")
st.markdown("""
<div class="card">
    <h3>About Growth Mindset</h3>
    <p>Embrace challenges, persist through obstacles, learn from criticism, 
    and find lessons in the success of others. Your abilities can be developed!</p>
</div>
""", unsafe_allow_html=True)

# Data Processing Functions
def load_data(file):
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        elif file.name.endswith('.xlsx'):
            return pd.read_excel(file)
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def show_stats(df):
    with st.expander("📈 Summary Statistics", expanded=False):
        st.dataframe(df.describe().style.background_gradient(cmap='Blues'))
    
    with st.expander("🔍 Data Preview", expanded=False):
        st.dataframe(df.head().style.set_properties(**{
            'background-color': '#1e2229',
            'color': 'white',
            'border': '1px solid #2a2f3b'
        }))

def visualize_data(df):
    st.subheader("📊 Data Visualization")
    
    col1, col2 = st.columns(2)
    with col1:
        chart_type = st.selectbox(
            "Select Chart Type",
            ["Bar", "Line", "Scatter", "Histogram", "Box", "Pie"],
            key="chart_type"
        )
    
    with col2:
        x_axis = st.selectbox("X-axis", df.columns, key="x_axis")
        y_axis = st.selectbox(
            "Y-axis", 
            df.columns, 
            disabled=chart_type in ["Histogram", "Pie"],
            key="y_axis"
        )
    
    try:
        if chart_type == "Bar":
            fig = px.bar(df, x=x_axis, y=y_axis, color=x_axis,
                        template="plotly_dark")
        elif chart_type == "Line":
            fig = px.line(df, x=x_axis, y=y_axis, color=x_axis,
                         template="plotly_dark")
        elif chart_type == "Scatter":
            fig = px.scatter(df, x=x_axis, y=y_axis, color=x_axis,
                           template="plotly_dark")
        elif chart_type == "Histogram":
            fig = px.histogram(df, x=x_axis, template="plotly_dark")
        elif chart_type == "Box":
            fig = px.box(df, x=x_axis, y=y_axis, template="plotly_dark")
        elif chart_type == "Pie":
            fig = px.pie(df, names=x_axis, template="plotly_dark")
        
        fig.update_layout({
            'plot_bgcolor': 'rgba(30, 34, 41, 1)',
            'paper_bgcolor': 'rgba(30, 34, 41, 1)',
        })
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")

def show_correlations(df):
    st.subheader("🧩 Correlation Matrix")
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 1:
        try:
            corr = numeric_df.corr()
            plt.figure(figsize=(10, 8), facecolor='#1e2229')
            sns.heatmap(corr, annot=True, cmap="Blues", 
                       annot_kws={"color": "white"})
            plt.xticks(color='white')
            plt.yticks(color='white')
            st.pyplot(plt)
            plt.clf()
        except Exception as e:
            st.error(f"Error creating correlation matrix: {str(e)}")
    else:
        st.warning("Not enough numeric columns for correlation analysis")

def download_data(df):
    st.subheader("💾 Export Data")
    col1, col2 = st.columns(2)
    with col1:
        format_type = st.radio(
            "Select Format", 
            ["CSV", "Excel"],
            key="export_format"
        )
    with col2:
        if format_type == "CSV":
            data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download CSV",
                data=data,
                file_name="growth_data.csv",
                mime="text/csv",
                key="csv_download"
            )
        else:
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            st.download_button(
                "Download Excel",
                data=output.getvalue(),
                file_name="growth_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="excel_download"
            )

# Main App Logic
if uploaded_file:
    df = load_data(uploaded_file)
    if df is not None:
        if user_name:
            st.success(f"Welcome, {user_name}! Let's analyze your growth data.")
        show_stats(df)
        visualize_data(df)
        show_correlations(df)
        download_data(df)
        st.balloons()
else:
    st.info("👆 Upload your data file to begin analysis")
    st.markdown("""
    <div style="text-align: center; margin-top: 20px;">
        <img src="https://via.placeholder.com/800x400/1e2229/4b6cb7?text=Upload+Your+Data+to+Begin" 
             style="border-radius: 10px; border: 1px solid #2a2f3b;" 
             width="100%">
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    Made with ❤️ by Farah Asghar | © 2025 | v1.0
</div>
""", unsafe_allow_html=True)