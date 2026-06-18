# app.py
# -----------------------------------------------------------------------------
# Streamlit Dashboard: BPSDM Jatim Live Chat Sentiment Analytics
# -----------------------------------------------------------------------------

import os
import json
import re
import datetime
from collections import Counter
from zoneinfo import ZoneInfo
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

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
        background-color: #0b0c10;
        color: #ffffff;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #12131c !important;
        border-right: 1px solid #222533;
    }
    
    /* Header fonts */
    h1, h2, h3 {
        font-family: 'Inter', 'Outfit', sans-serif;
        font-weight: 700;
        color: #ffffff;
    }
    
    /* Custom spacing and styles for native Streamlit metrics */
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    div[data-testid="metric-container"] label {
        color: #8b92b6 !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.8rem !important;
    }
    
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# DYNAMIC TIMESTAMP HELPER
# -----------------------------------------------------------------------------
def get_last_updated_time():
    local_csv_path = "bpsdm_gemini_sentiment_results.csv"
    if os.path.exists(local_csv_path):
        mtime = os.path.getmtime(local_csv_path)
        dt = datetime.datetime.fromtimestamp(mtime, tz=ZoneInfo('Asia/Jakarta'))
        return dt.strftime("%A, %d %B %Y, %H:%M WIB")
    
    # Fallback calculation if file info is missing
    now = datetime.datetime.now(ZoneInfo('Asia/Jakarta'))
    days_to_last_thursday = (now.weekday() - 3) % 7
    last_thursday = now - datetime.timedelta(days=days_to_last_thursday)
    if now.weekday() == 3 and now.hour < 13:
        last_thursday = last_thursday - datetime.timedelta(days=7)
    last_thursday_13 = last_thursday.replace(hour=13, minute=0, second=0, microsecond=0)
    return last_thursday_13.strftime("%A, %d %B %Y, 13:00 WIB")

# -----------------------------------------------------------------------------
# DATA LOADING FUNCTION (Google Sheets with local fallback)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=600) # Cache data for 10 minutes to prevent API spam
def load_data():
    creds_json = os.environ.get("GOOGLE_SHEETS_CREDENTIALS")
    spreadsheet_name = "BPSDM Jatim Live Chat Sentiments"
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
                st.sidebar.success("⚡ Loaded live from Google Sheets Cloud!")
                return df
        except Exception as e:
            st.sidebar.warning(f"Could not connect to Google Sheets ({e}). Using local fallback...")
            
    # Fallback to local CSV file
    if os.path.exists(local_csv_path):
        df = pd.read_csv(local_csv_path)
        st.sidebar.info("📂 Running on offline cached backup (Local CSV).")
        return df
    else:
        st.sidebar.error("❌ No data source found. Execute the ETL pipeline to generate results.")
        return pd.DataFrame()

df = load_data()

# Ensure required columns exist
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
# TIME FORMATTING HELPER
# -----------------------------------------------------------------------------
def format_time_offset(seconds):
    try:
        val = float(seconds)
        mins = int(val // 60)
        secs = int(val % 60)
        return f"{mins:02d}:{secs:02d}"
    except Exception:
        return "00:00"

# -----------------------------------------------------------------------------
# STOPWORDS & WORD FREQUENCY CALCULATOR
# -----------------------------------------------------------------------------
INDONESIAN_STOPWORDS = {
    'yang', 'di', 'dan', 'ke', 'dari', 'ini', 'itu', 'ada', 'untuk', 'with', 'dengan', 'saya', 'kami', 
    'kita', 'pak', 'bu', 'ya', 'kok', 'aja', 'saja', 'juga', 'lah', 'kah', 'pun', 'adalah', 
    'bisa', 'akan', 'tidak', 'ga', 'gak', 'bukan', 'tapi', 'namun', 'oleh', 'karena', 'sehingga',
    'sudah', 'telah', 'belum', 'sedang', 'akan', 'dalam', 'pada', 'atau', 'serta', 'jika', 
    'kalau', 'maka', 'tentang', 'seperti', 'kamu', 'dia', 'mereka', 'siap', 'selamat', 'pagi', 
    'salam', 'assalamualaikum', 'hadir', 'mengikuti', 'nyimak', 'absen', 'presensi', 'link', 
    'sertifikat', 'dinas', 'prov', 'kab', 'kec', 'masuk', 'terima', 'kasih', 'terimakasih', 
    'mohon', 'ijin', 'izin', 'info', 'mase', 'mbak', 'bang', 'kak'
}

def get_word_frequencies(messages):
    words = []
    for msg in messages:
        tokens = re.findall(r'\b\w+\b', str(msg).lower())
        for token in tokens:
            if token not in INDONESIAN_STOPWORDS and len(token) > 2:
                words.append(token)
    return Counter(words).most_common(15)

# -----------------------------------------------------------------------------
# SIDEBAR CONTENT (Metadata Context & Filters)
# -----------------------------------------------------------------------------
st.sidebar.title("🏛️ BPSDM Jatim Control Panel")

last_updated = get_last_updated_time()
st.sidebar.markdown(f"""
**📅 System Status:**
*   **Last ETL Sync:**
    `{last_updated}`
*   *Updated every Thursday at 13:00 WIB*
""")

if not df.empty:
    # 1. Video Selector Filter
    titles = ["All Videos"] + list(df["Video Title"].unique())
    selected_title = st.sidebar.selectbox("🎯 Select Webinar Video:", titles)
    
    # Apply filter based on selection
    if selected_title != "All Videos":
        filtered_df = df[df["Video Title"] == selected_title]
    else:
        filtered_df = df

    # 2. Metadata Information Block
    st.sidebar.write("---")
    st.sidebar.markdown("### 📽️ Processed Webinar Metadata")
    
    # Get the 3 unique webinars from the active dataframe
    webinar_metadata = df[["Video Title", "Video ID"]].drop_duplicates().head(3)
    for index, row in webinar_metadata.iterrows():
        title_truncated = row["Video Title"][:50] + "..." if len(row["Video Title"]) > 50 else row["Video Title"]
        st.sidebar.markdown(f"""
        **🎬 {title_truncated}**
        *   **ID:** `{row['Video ID']}`
        *   **Link:** [Open YouTube](https://youtube.com/watch?v={row['Video ID']})
        """)
        
    # -------------------------------------------------------------------------
    # MAIN DASHBOARD PAGE
    # -------------------------------------------------------------------------
    st.title("🏛️ YouTube Live Chat Sentiment Dashboard")
    st.subheader("BPSDM Provinsi Jawa Timur — Automated Analytics Portal")
    st.markdown(f"*Data automatically updated on: **{last_updated}***")
    
    # -------------------------------------------------------------------------
    # SECTION 1: KEY PERFORMANCE INDICATORS (KPIs)
    # -------------------------------------------------------------------------
    st.write("### 📊 Key Performance Indicators (KPIs)")
    total_chats = len(filtered_df)
    
    # Sentiment calculations
    counts = filtered_df["Sentiment"].value_counts()
    pos_count = counts.get("Positive", 0)
    neg_count = counts.get("Negative", 0)
    
    pos_pct = (pos_count / total_chats * 100) if total_chats > 0 else 0
    neg_pct = (neg_count / total_chats * 100) if total_chats > 0 else 0
    
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    with col_kpi1:
        st.metric(
            label="Total Chats Analyzed", 
            value=f"{total_chats:,}",
            help="Total number of clean chat lines successfully processed from the active webinars."
        )
    with col_kpi2:
        st.metric(
            label="Apparatus Satisfaction Rate", 
            value=f"{pos_pct:.1f}%",
            help="Percentage of chats classified as Positive (representing satisfaction, praise, or appreciation)."
        )
    with col_kpi3:
        st.metric(
            label="Complaint Rate", 
            value=f"{neg_pct:.1f}%",
            help="Percentage of chats classified as Negative (representing technical issues, complaints, or registration problems)."
        )
        
    st.write("---")
    
    # -------------------------------------------------------------------------
    # SECTION 2: SENTIMENT DISTRIBUTION AND TREND CHARTS
    # -------------------------------------------------------------------------
    st.write("### 🍰 Sentiment Distribution & Trend Chart")
    
    col_vis_left, col_vis_right = st.columns([1, 1])
    
    with col_vis_left:
        st.write("#### 🍩 Sentiment Composition (Overall)")
        if total_chats > 0:
            fig_donut = px.pie(
                filtered_df,
                names="Sentiment",
                color="Sentiment",
                color_discrete_map={"Positive": "#98c379", "Negative": "#e06c75", "Neutral": "#abb2bf"},
                hole=0.45
            )
            fig_donut.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(font=dict(color="#ffffff")),
                margin=dict(t=20, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_donut, use_container_width=True)
        else:
            st.info("No data available for the active selection.")
            
    with col_vis_right:
        st.write("#### 📊 Sentiment Portion Comparison per Webinar")
        if total_chats > 0:
            # Group by Video Title and Sentiment, calculate percentage
            comp_df = filtered_df.groupby(["Video Title", "Sentiment"]).size().reset_index(name="Count")
            comp_df["Percentage"] = comp_df.groupby("Video Title")["Count"].transform(lambda x: (x / x.sum()) * 100)
            
            fig_comp = px.bar(
                comp_df,
                x="Video Title",
                y="Percentage",
                color="Sentiment",
                color_discrete_map={"Positive": "#98c379", "Negative": "#e06c75", "Neutral": "#abb2bf"},
                barmode="group",
                labels={"Video Title": "Webinar Title", "Percentage": "Percentage Portion (%)"}
            )
            fig_comp.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#ffffff"),
                xaxis=dict(showgrid=False, title=None, tickfont=dict(size=10)),
                yaxis=dict(showgrid=True, gridcolor="#222533", range=[0, 100]),
                legend=dict(font=dict(color="#ffffff")),
                margin=dict(t=20, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_comp, use_container_width=True)
        else:
            st.info("No data available for the active selection.")
            
    # Timeline Trend Analysis
    st.write("#### 📈 Webinar Timeline Sentiment Trend (5-Minute Windows)")
    if total_chats > 0:
        timeline_df = filtered_df.copy()
        # Convert seconds to 5-minute interval blocks
        timeline_df["Minute"] = (timeline_df["Waktu"] // 300) * 5
        
        trend_df = timeline_df.groupby(["Minute", "Sentiment"]).size().reset_index(name="Count")
        
        # Checkbox to isolate and view only complaints/negative trends
        isolate_complaints = st.checkbox("🔍 Isolate and display only Negative Sentiment (Complaints) timeline")
        if isolate_complaints:
            trend_df = trend_df[trend_df["Sentiment"] == "Negative"]
            color_map = {"Negative": "#e06c75"}
        else:
            color_map = {"Positive": "#98c379", "Negative": "#e06c75", "Neutral": "#abb2bf"}
            
        fig_line = px.line(
            trend_df,
            x="Minute",
            y="Count",
            color="Sentiment",
            color_discrete_map=color_map,
            labels={"Minute": "Webinar Elapsed Time (Minutes)", "Count": "Message Count"},
            markers=True
        )
        fig_line.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor="#222533", title_font=dict(color="#ffffff"), tickfont=dict(color="#ffffff")),
            yaxis=dict(showgrid=True, gridcolor="#222533", title_font=dict(color="#ffffff"), tickfont=dict(color="#ffffff")),
            legend=dict(font=dict(color="#ffffff")),
            margin=dict(t=10, b=10, l=10, r=10)
        )
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("No data available for the active selection.")
        
    st.write("---")
    
    # -------------------------------------------------------------------------
    # SECTION 3: PPID CONTROL ROOM (COMPLAINT ANALYSIS)
    # -------------------------------------------------------------------------
    st.write("### 🚨 PPID Control Room: Complaint Analysis & Action Portal")
    
    col_ctrl_left, col_ctrl_right = st.columns([2, 3])
    
    # Filter dataset for negative complaints
    negative_df = filtered_df[filtered_df["Sentiment"] == "Negative"].copy()
    
    with col_ctrl_left:
        st.write("#### 🏷️ Top Obstacles / Keywords in Negative Sentiment")
        if not negative_df.empty:
            neg_words = get_word_frequencies(negative_df["Pesan"])
            if neg_words:
                w_df = pd.DataFrame(neg_words, columns=["Keyword", "Frequency"])
                fig_neg_bar = px.bar(
                    w_df,
                    x="Frequency",
                    y="Keyword",
                    orientation="h",
                    color="Frequency",
                    color_continuous_scale=px.colors.sequential.Reds
                )
                fig_neg_bar.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=False, tickfont=dict(color="#ffffff")),
                    yaxis=dict(showgrid=False, tickfont=dict(color="#ffffff"), categoryorder="total ascending"),
                    coloraxis_showscale=False,
                    margin=dict(t=10, b=10, l=10, r=10)
                )
                st.plotly_chart(fig_neg_bar, use_container_width=True)
            else:
                st.info("Not enough keyword volume to parse obstacles.")
        else:
            st.success("🎉 No negative feedback found for this selection!")
            
    with col_ctrl_right:
        st.write("#### 📋 Urgent Complaints & Feedback Table (Requires PPID Attention)")
        if not negative_df.empty:
            # Format time offset to readable MM:SS
            negative_df["Timestamp"] = negative_df["Waktu"].apply(format_time_offset)
            
            # Display latest/most urgent 10 negative chats
            display_cols = ["Timestamp", "Username", "Pesan", "Video Title"]
            st.dataframe(
                negative_df[display_cols].head(10),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.success("🎉 Keep up the good work! No complaints require attention.")
            
    st.write("---")
    
    # -------------------------------------------------------------------------
    # SECTION 4: CONVERSATIONAL SEARCH RECORDS
    # -------------------------------------------------------------------------
    st.write("### 🔍 Search Conversational Records")
    search_query = st.text_input("Enter keywords to search chat database (e.g. 'suara', 'materi', 'pemateri'):")
    
    display_df = filtered_df.copy()
    if search_query:
        display_df = display_df[display_df["Pesan"].str.contains(search_query, case=False, na=False)]
        
    display_df["Timestamp"] = display_df["Waktu"].apply(format_time_offset)
    st.dataframe(
        display_df[["Timestamp", "Username", "Pesan", "Sentiment", "Video Title"]],
        use_container_width=True,
        hide_index=True
    )
    
else:
    st.info("Please verify data source configuration. No records loaded.")
