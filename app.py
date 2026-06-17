# app.py
# -----------------------------------------------------------------------------
# Streamlit Dashboard: BPSDM Jatim Live Chat Sentiment Analytics
# -----------------------------------------------------------------------------

import os
import json
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re

# Set page layout to wide and add title
st.set_page_config(
    page_title="BPSDM Jatim Sentiment Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# CUSTOM CSS STYLE INJECTIONS (Premium Dark / Glassmorphism Vibe)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Main body background color */
    .stApp {
        background-color: #0f111a;
        color: #ffffff;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #161925 !important;
        border-right: 1px solid #2e3440;
    }
    
    /* Header fonts */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        color: #ffffff;
    }
    
    /* Glassmorphic Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .metric-val {
        font-size: 2rem;
        font-weight: 800;
        margin: 5px 0;
    }
    .metric-lbl {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #8888aa;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# DATA LOADING FUNCTION (Google Sheets with local fallback)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=600) # Cache data for 10 minutes to prevent API spam
def load_data():
    creds_json = os.environ.get("GOOGLE_SHEETS_CREDENTIALS")
    spreadsheet_name = "BPSDM Jatim Live Chat Sentiments"
    sheet_name = "Sentiment Data"
    local_csv_path = "bpsdm_gemini_sentiment_results.csv"
    
    # Attempt to load from Google Sheets
    if creds_json:
        try:
            import gspread
            from google.oauth2.service_account import Credentials
            
            creds_dict = json.loads(creds_json)
            scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
            creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
            client = gspread.authorize(creds)
            
            sh = client.open(spreadsheet_name)
            worksheet = sh.get_worksheet(0)
            records = worksheet.get_all_records()
            if records:
                df = pd.DataFrame(records)
                st.sidebar.success("⚡ Loaded data dynamically from Google Sheets Cloud Database!")
                return df
        except Exception as e:
            st.sidebar.warning(f"Could not connect to Google Sheets Cloud ({e}). Falling back to local dataset...")
            
    # Fallback to local CSV file
    if os.path.exists(local_csv_path):
        df = pd.read_csv(local_csv_path)
        st.sidebar.info("📂 Running on offline cached backup (Local CSV).")
        return df
    else:
        st.sidebar.error("❌ No data source found. Please execute the ETL pipeline to generate results.")
        return pd.DataFrame()

df = load_data()

# Ensure fallback columns exist if loading from old CSV files
if not df.empty:
    required_cols = {
        "Waktu": 0,
        "Username": "Unknown",
        "Pesan": "",
        "Sentiment": "Neutral",
        "Video ID": "Unknown",
        "Video Title": "Webinar Live Chat"
    }
    for col, default_val in required_cols.items():
        if col not in df.columns:
            df[col] = default_val


# -----------------------------------------------------------------------------
# STOPWORDS & WORD CLOUD CALCULATOR
# -----------------------------------------------------------------------------
INDONESIAN_STOPWORDS = {
    'yang', 'di', 'dan', 'ke', 'dari', 'ini', 'itu', 'ada', 'untuk', 'dengan', 'saya', 'kami', 
    'kita', 'pak', 'bu', 'ya', 'kok', 'aja', 'saja', 'juga', 'lah', 'kah', 'pun', 'adalah', 
    'bisa', 'akan', 'tidak', 'ga', 'gak', 'bukan', 'tapi', 'namun', 'oleh', 'karena', 'sehingga',
    'sudah', 'telah', 'belum', 'sedang', 'akan', 'dalam', 'pada', 'atau', 'serta', 'jika', 
    'kalau', 'maka', 'tentang', 'seperti', 'kamu', 'dia', 'mereka', 'siap', 'selamat', 'pagi', 
    'salam', 'assalamualaikum', 'hadir', 'mengikuti', 'nyimak', 'absen', 'presensi', 'link', 
    'sertifikat', 'dinas', 'prov', 'kab', 'kec', 'masuk'
}

def get_word_frequencies(messages):
    words = []
    for msg in messages:
        # Tokenize words using regex, convert to lowercase
        tokens = re.findall(r'\b\w+\b', str(msg).lower())
        for token in tokens:
            if token not in INDONESIAN_STOPWORDS and len(token) > 2:
                words.append(token)
    return Counter(words).most_common(20)

# -----------------------------------------------------------------------------
# SIDEBAR FILTERS
# -----------------------------------------------------------------------------
st.sidebar.title("📊 Filter Dashboard")

if not df.empty:
    # Video selector filter
    titles = ["All Videos"] + list(df["Video Title"].unique())
    selected_title = st.sidebar.selectbox("Select Livestream Video:", titles)
    
    # Filter dataset based on selection
    if selected_title != "All Videos":
        filtered_df = df[df["Video Title"] == selected_title]
    else:
        filtered_df = df
        
    # Sentiment quick selector filter
    sentiments = ["All Sentiments"] + list(df["Sentiment"].unique())
    selected_sentiment = st.sidebar.selectbox("Filter Sentiment:", sentiments)
    
    if selected_sentiment != "All Sentiments":
        filtered_df = filtered_df[filtered_df["Sentiment"] == selected_sentiment]
        
    # Render layout
    st.title("🏛️ YouTube Live Chat Sentiment Dashboard")
    st.subheader("BPSDM Provinsi Jawa Timur — Periodic ETL Results")
    st.write("Provides automatic cleansing and Gemini AI analysis of public comments from recent live webinar sessions.")
    
    # -------------------------------------------------------------------------
    # ROW 1: KPI METRIC CARDS
    # -------------------------------------------------------------------------
    total_chats = len(filtered_df)
    
    # Count sentiments
    counts = filtered_df["Sentiment"].value_counts()
    pos_count = counts.get("Positive", 0)
    neg_count = counts.get("Negative", 0)
    neu_count = counts.get("Neutral", 0)
    
    # Percentages
    pos_pct = (pos_count / total_chats * 100) if total_chats > 0 else 0
    neg_pct = (neg_count / total_chats * 100) if total_chats > 0 else 0
    neu_pct = (neu_count / total_chats * 100) if total_chats > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">Total Chats Cleansed</div>
            <div class="metric-val" style="color: #61afef;">{total_chats}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">Positive Sentiments</div>
            <div class="metric-val" style="color: #98c379;">{pos_count} <span style="font-size:1.1rem; font-weight:400;">({pos_pct:.1f}%)</span></div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">Negative Sentiments</div>
            <div class="metric-val" style="color: #e06c75;">{neg_count} <span style="font-size:1.1rem; font-weight:400;">({neg_pct:.1f}%)</span></div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">Neutral Sentiments</div>
            <div class="metric-val" style="color: #abb2bf;">{neu_count} <span style="font-size:1.1rem; font-weight:400;">({neu_pct:.1f}%)</span></div>
        </div>
        """, unsafe_allow_html=True)
        
    st.write("---")
    
    # -------------------------------------------------------------------------
    # ROW 2: SENTIMENT DISTRIBUTION & TIMELINE PLOTS
    # -------------------------------------------------------------------------
    col_left, col_right = st.columns([2, 3])
    
    with col_left:
        st.write("### 🍰 Sentiment Distribution")
        if total_chats > 0:
            fig_pie = px.pie(
                names=["Positive", "Negative", "Neutral"],
                values=[pos_count, neg_count, neu_count],
                color=["Positive", "Negative", "Neutral"],
                color_discrete_map={"Positive": "#98c379", "Negative": "#e06c75", "Neutral": "#abb2bf"},
                hole=0.4
            )
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(font=dict(color="#ffffff")),
                margin=dict(t=10, b=10, l=10, r=10)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No data matching filters.")
            
    with col_right:
        st.write("### 📈 Webinar Timeline Sentiment Trend")
        if total_chats > 0:
            # Group timeline into 5-minute segments to smooth line trends
            timeline_df = filtered_df.copy()
            # Convert time offset in seconds to minutes
            timeline_df["Minute"] = (timeline_df["Waktu"] // 300) * 5 # 5-min intervals
            
            trend_df = timeline_df.groupby(["Minute", "Sentiment"]).size().reset_index(name="Count")
            
            fig_line = px.line(
                trend_df,
                x="Minute",
                y="Count",
                color="Sentiment",
                color_discrete_map={"Positive": "#98c379", "Negative": "#e06c75", "Neutral": "#abb2bf"},
                labels={"Minute": "Webinar Elapsed Time (Minutes)", "Count": "Messages Count"},
                markers=True
            )
            fig_line.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=True, gridcolor="#2e3440", title_font=dict(color="#ffffff"), tickfont=dict(color="#ffffff")),
                yaxis=dict(showgrid=True, gridcolor="#2e3440", title_font=dict(color="#ffffff"), tickfont=dict(color="#ffffff")),
                legend=dict(font=dict(color="#ffffff")),
                margin=dict(t=10, b=10, l=10, r=10)
            )
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("No data matching filters.")
            
    st.write("---")
    
    # -------------------------------------------------------------------------
    # ROW 3: TOP KEYWORDS & SEARCH DATA TABLE
    # -------------------------------------------------------------------------
    col_left_word, col_right_table = st.columns([2, 3])
    
    with col_left_word:
        st.write("### 🏷️ Top Key Phrases (Excluding Stopwords)")
        if total_chats > 0:
            word_freqs = get_word_frequencies(filtered_df["Pesan"])
            if word_freqs:
                w_df = pd.DataFrame(word_freqs, columns=["Word", "Frequency"])
                fig_bar = px.bar(
                    w_df,
                    x="Frequency",
                    y="Word",
                    orientation="h",
                    color="Frequency",
                    color_continuous_scale=px.colors.sequential.Sunset_r
                )
                fig_bar.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=False, tickfont=dict(color="#ffffff")),
                    yaxis=dict(showgrid=False, tickfont=dict(color="#ffffff"), categoryorder="total ascending"),
                    coloraxis_showscale=False,
                    margin=dict(t=10, b=10, l=10, r=10)
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("Not enough keywords to plot.")
        else:
            st.info("No data matching filters.")
            
    with col_right_table:
        st.write("### 🔍 Search Conversational Records")
        search_query = st.text_input("Enter search keywords (e.g. 'narasum', 'link', 'materi'):")
        
        display_df = filtered_df.copy()
        if search_query:
            display_df = display_df[display_df["Pesan"].str.contains(search_query, case=False, na=False)]
            
        st.dataframe(
            display_df[["Waktu", "Username", "Pesan", "Sentiment"]],
            use_container_width=True,
            hide_index=True
        )

else:
    st.info("Please verify data source configuration. No records loaded.")
