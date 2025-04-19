import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime

# ========== PAGE CONFIGURATION ==========
st.set_page_config(
    page_title="Growth Mindset Data Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CUSTOM THEME ==========
def apply_custom_theme():
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
            border-right: 1px solid #2a2f3b !important;
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
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #3a5a9b !important;
            transform: scale(1.02);
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
        .stExpander > div > div {
            background-color: #1e2229 !important;
        }
        
        /* Footer */
        .footer {
            background-color: #1a1d24 !important;
            color: white !important;
            padding: 15px;
            text-align: center;
            border-top: 1px solid #2a2f3b;
            font-size: 0.9rem;
        }
        
        /* Plotly chart background */
        .js-plotly-plot .plotly {
            background-color: #1e2229 !important;
        }
        
        /* Tooltips */
        .stTooltip {
            background-color: #2a2f3b !important;
            color: white !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Apply dark theme to matplotlib and seaborn
    plt.style.use('dark_background')
    sns.set_style("darkgrid")
    sns.set_palette("muted")

apply_custom_theme()

# ========== SIDEBAR ==========
def create_sidebar():
    with st.sidebar:
        st.title("🧠 Growth Mindset Analyzer")
        st.markdown("**Develop your abilities through dedication and hard work!**")
        
        # User information
        with st.container():
            user_name = st.text_input(
                "Your Name", 
                placeholder="Enter your name", 
                key="name_input"
            )
            
            uploaded_file = st.file_uploader(
                "Upload Your Data (CSV/Excel)",
                type=["csv", "xlsx"],
                help="Maximum file size: 200MB",
                key="file_uploader"
            )
        
        # App information
        with st.expander("ℹ️ About This App"):
            st.markdown("""
            This app helps you analyze data through the lens of growth mindset principles.
            
            **Features:**
            - Data cleaning and preprocessing
            - Interactive visualizations
            - Statistical analysis
            - Correlation insights
            """)
        
        return user_name, uploaded_file

# ========== DATA PROCESSING FUNCTIONS ==========
def load_data(file):
    """Load data from uploaded file with error handling"""
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        elif file.name.endswith('.xlsx'):
            return pd.read_excel(file)
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        return None

def clean_data(df):
    """Basic data cleaning operations"""
    with st.expander("🧹 Data Cleaning Options", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Remove Duplicates"):
                initial = len(df)
                df = df.drop_duplicates()
                st.success(f"Removed {initial - len(df)} duplicate rows")
        
        with col2:
            if st.button("Fill Missing Values"):
                numeric_cols = df.select_dtypes(include=np.number).columns
                if not numeric_cols.empty:
                    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                    st.success("Filled missing numeric values with mean")
                else:
                    st.warning("No numeric columns found")
        
        with col3:
            if st.button("Drop Empty Rows"):
                initial = len(df)
                df = df.dropna()
                st.success(f"Dropped {initial - len(df)} empty rows")
    
    return df

def show_data_stats(df):
    """Display data statistics and preview"""
    with st.expander("📊 Data Overview", expanded=True):
        st.subheader("Basic Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Rows", len(df))
            st.metric("Total Columns", len(df.columns))
        
        with col2:
            st.metric("Numeric Columns", len(df.select_dtypes(include=np.number).columns))
            st.metric("Categorical Columns", len(df.select_dtypes(exclude=np.number).columns))
        
        st.subheader("Summary Statistics")
        st.dataframe(df.describe().style.background_gradient(cmap='Blues'))
    
    with st.expander("🔍 Data Preview", expanded=False):
        st.dataframe(df.head(10).style.set_properties(**{
            'background-color': '#1e2229',
            'color': 'white',
            'border': '1px solid #2a2f3b'
        })

def visualize_data(df):
    """Interactive data visualization"""
    st.subheader("📈 Data Visualization")
    
    # Chart selection
    col1, col2 = st.columns([1, 2])
    with col1:
        chart_type = st.selectbox(
            "Select Chart Type",
            ["Bar", "Line", "Scatter", "Histogram", "Box", "Pie", "Violin"],
            key="chart_type"
        )
        
        x_axis = st.selectbox("X-axis", df.columns, key="x_axis")
        y_axis = st.selectbox(
            "Y-axis", 
            df.columns, 
            disabled=chart_type in ["Histogram", "Pie"],
            key="y_axis"
        )
        
        color_by = st.selectbox(
            "Color By",
            ["None"] + list(df.columns),
            key="color_by"
        )
    
    # Generate chart
    try:
        if chart_type == "Bar":
            fig = px.bar(
                df, 
                x=x_axis, 
                y=y_axis, 
                color=None if color_by == "None" else color_by,
                template="plotly_dark"
            )
        elif chart_type == "Line":
            fig = px.line(
                df, 
                x=x_axis, 
                y=y_axis, 
                color=None if color_by == "None" else color_by,
                template="plotly_dark"
            )
        elif chart_type == "Scatter":
            fig = px.scatter(
                df, 
                x=x_axis, 
                y=y_axis, 
                color=None if color_by == "None" else color_by,
                template="plotly_dark"
            )
        elif chart_type == "Histogram":
            fig = px.histogram(
                df, 
                x=x_axis, 
                color=None if color_by == "None" else color_by,
                template="plotly_dark"
            )
        elif chart_type == "Box":
            fig = px.box(
                df, 
                x=x_axis, 
                y=y_axis, 
                color=None if color_by == "None" else color_by,
                template="plotly_dark"
            )
        elif chart_type == "Pie":
            fig = px.pie(
                df, 
                names=x_axis, 
                template="plotly_dark"
            )
        elif chart_type == "Violin":
            fig = px.violin(
                df,
                x=x_axis,
                y=y_axis,
                color=None if color_by == "None" else color_by,
                template="plotly_dark"
            )
        
        fig.update_layout({
            'plot_bgcolor': 'rgba(30, 34, 41, 1)',
            'paper_bgcolor': 'rgba(30, 34, 41, 1)',
            'font': {'color': 'white'}
        })
        
        with col2:
            st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ Error creating visualization: {str(e)}")

def show_correlations(df):
    """Display correlation matrix"""
    st.subheader("🧩 Correlation Matrix")
    
    numeric_df = df.select_dtypes(include=np.number)
    if len(numeric_df.columns) > 1:
        try:
            corr = numeric_df.corr()
            
            plt.figure(figsize=(10, 8), facecolor='#1e2229')
            sns.heatmap(
                corr, 
                annot=True, 
                cmap="Blues", 
                annot_kws={"color": "white"},
                linewidths=0.5
            )
            plt.xticks(color='white', rotation=45)
            plt.yticks(color='white')
            st.pyplot(plt)
            plt.clf()
        except Exception as e:
            st.error(f"❌ Error creating correlation matrix: {str(e)}")
    else:
        st.warning("⚠️ Not enough numeric columns for correlation analysis")

def export_data(df):
    """Data export functionality"""
    st.subheader("💾 Export Data")
    
    col1, col2 = st.columns(2)
    with col1:
        format_type = st.radio(
            "Select Format", 
            ["CSV", "Excel"],
            key="export_format"
        )
        
        filename = st.text_input(
            "File Name",
            value=f"growth_data_{datetime.now().strftime('%Y%m%d')}",
            key="filename"
        )
    
    with col2:
        if format_type == "CSV":
            data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download CSV",
                data=data,
                file_name=f"{filename}.csv",
                mime="text/csv",
                key="csv_download",
                help="Download data as CSV file"
            )
        else:
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name="Data")
            st.download_button(
                "Download Excel",
                data=output.getvalue(),
                file_name=f"{filename}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="excel_download",
                help="Download data as Excel file"
            )

# ========== MAIN APP LOGIC ==========
def main():
    # Header section
    st.title("📊 Growth Mindset Data Analyzer")
    st.markdown("""
    <div class="card">
        <h3>About Growth Mindset</h3>
        <p>Embrace challenges, persist through obstacles, learn from criticism, 
        and find lessons in the success of others. Your abilities can be developed!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get user input from sidebar
    user_name, uploaded_file = create_sidebar()
    
    # Main content
    if uploaded_file:
        df = load_data(uploaded_file)
        
        if df is not None:
            if user_name:
                st.success(f"👋 Welcome, {user_name}! Let's analyze your growth data.")
            
            # Data processing pipeline
            df = clean_data(df)
            show_data_stats(df)
            visualize_data(df)
            show_correlations(df)
            export_data(df)
            
            # Celebration
            st.balloons()
    else:
        st.info("📤 Please upload your data file to begin analysis")
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
        Made with Streamlit❤️ by Farah Asghar | © 2025 | v1.1
    </div>
    """.format(year=datetime.now().year), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
