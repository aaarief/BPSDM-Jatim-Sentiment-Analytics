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
# LANGUAGE SELECTOR & TRANSLATION DICTIONARY
# -----------------------------------------------------------------------------
lang_choice = st.sidebar.selectbox("🌐 Pilih Bahasa / Select Language", ["Bahasa Indonesia", "English"])

if lang_choice == "Bahasa Indonesia":
    t = {
        "title": "Dashboard Analisis Sentimen Live Chat YouTube",
        "subtitle": "BPSDM Provinsi Jawa Timur",
        "last_updated_prefix": "Data diperbarui otomatis pada: **",
        "status_title": "Status Sistem:",
        "last_sync": "Sinkronisasi ETL Terakhir",
        "updated_schedule": "Diperbarui setiap Kamis pukul 13:00 WIB",
        "select_webinar": "Pilih Video Webinar:",
        "all_videos": "Semua Video",
        "metadata_header": "Metadata Webinar Terproses",
        "open_yt": "Buka YouTube",
        "kpi_header": "Indikator Kinerja Utama (KPI)",
        "kpi1_label": "Total Diskusi Aktif",
        "kpi1_help": "Jumlah baris chat aktif (tidak termasuk spam absen/salam) yang berhasil diproses.",
        "kpi_present_label": "Kehadiran Peserta",
        "kpi_present_help": "Jumlah baris chat yang diidentifikasi sebagai absen presensi atau salam formal.",
        "kpi2_label": "Kepuasan Aparatur",
        "kpi2_help": "Persentase chat berkategori Positif (menunjukkan kepuasan, pujian, atau apresiasi) dari total diskusi aktif.",
        "kpi3_label": "Tingkat Keluhan",
        "kpi3_help": "Persentase chat berkategori Negatif (menunjukkan masalah teknis, keluhan, atau kendala pendaftaran) dari total diskusi aktif.",
        "filter_spam_toggle": "Saring Out Spam Absen / Salam",
        "charts_header": " Distribusi & Tren Sentimen",
        "donut_header": " Komposisi Sentimen (Keseluruhan)",
        "bar_header": " Perbandingan Porsi Sentimen per Webinar",
        "bar_x": "Judul Webinar",
        "bar_y": "Porsi Persentase (%)",
        "timeline_header": "📈 Tren Sentimen Linimasa Webinar (Jendela 5 Menit)",
        "timeline_toggle": "🔍 Isolasikan dan tampilkan hanya linimasa Sentimen Negatif (Keluhan)",
        "timeline_x": "Waktu Berjalan Webinar (Menit)",
        "timeline_y": "Jumlah Pesan",
        "no_data": "Data tidak tersedia untuk pilihan ini.",
        "ppid_header": "Analisis Keluhan & Aksi Tindak Lanjut",
        "obstacles_header": " Kata Kunci / Hambatan Utama pada Sentimen Negatif",
        "no_negatives": "Tidak ada keluhan yang ditemukan untuk pilihan ini!",
        "not_enough_keywords": "Volume kata kunci tidak cukup untuk dianalisis.",
        "urgent_table_header": "📋 Tabel Keluhan dan Masukan",
        "no_complaints_success": "Tidak ada keluhan yang membutuhkan perhatian.",
        "search_header": "🔍 Cari Riwayat Percakapan",
        "search_input_label": "Masukkan kata kunci untuk mencari database chat (contoh: 'suara', 'materi', 'pemateri'):",
        "col_timestamp": "Waktu",
        "col_username": "Nama Pengguna",
        "col_message": "Pesan",
        "col_sentiment": "Sentimen",
        "col_title": "Judul Video",
        "system_status": "Status Sistem"
    }
    sentiment_map = {
        "Positive": "Positif", 
        "Negative": "Negatif", 
        "Neutral": "Diskusi Netral",
        "Neutral - Discussion": "Diskusi Netral",
        "Attendance / Greeting": "Presensi / Salam"
    }
    color_map = {
        "Positif": "#98c379", 
        "Negatif": "#e06c75", 
        "Diskusi Netral": "#abb2bf",
        "Presensi / Salam": "#e5c07b"
    }
else:
    t = {
        "title": "YouTube Live Chat Sentiment Dashboard",
        "subtitle": "BPSDM Provinsi Jawa Timur",
        "last_updated_prefix": "Data automatically updated on: **",
        "status_title": "System Status:",
        "last_sync": "Last ETL Sync",
        "updated_schedule": "Updated every Thursday at 13:00 WIB",
        "select_webinar": "Select Webinar Video:",
        "all_videos": "All Videos",
        "metadata_header": " Processed Webinar Metadata",
        "open_yt": "Open YouTube",
        "kpi_header": "Key Performance Indicators (KPIs)",
        "kpi1_label": "Total Active Chats",
        "kpi1_help": "Total number of active chat lines (excluding attendance/greetings spam) processed.",
        "kpi_present_label": "Total Participants Present",
        "kpi_present_help": "Total number of chat entries identified as attendance check-ins or formal greetings.",
        "kpi2_label": "Apparatus Satisfaction Rate",
        "kpi2_help": "Percentage of chats classified as Positive (representing satisfaction, praise, or appreciation) out of active chats.",
        "kpi3_label": "Complaint Rate",
        "kpi3_help": "Percentage of chats classified as Negative (representing technical issues, complaints, or registration problems) out of active chats.",
        "filter_spam_toggle": "Filter Out Attendance Spam",
        "charts_header": " Sentiment Distribution & Trend Chart",
        "donut_header": " Sentiment Composition (Overall)",
        "bar_header": " Sentiment Portion Comparison per Webinar",
        "bar_x": "Webinar Title",
        "bar_y": "Percentage Portion (%)",
        "timeline_header": " Webinar Timeline Sentiment Trend (5-Minute Windows)",
        "timeline_toggle": "🔍 Isolate and display only Negative Sentiment (Complaints) timeline",
        "timeline_x": "Webinar Elapsed Time (Minutes)",
        "timeline_y": "Message Count",
        "no_data": "No data available for the active selection.",
        "ppid_header": "Complaint Analysis & Action Portal",
        "obstacles_header": "Top Obstacles / Keywords in Negative Sentiment",
        "no_negatives": "No negative feedback found for this selection!",
        "not_enough_keywords": "Not enough keyword volume to parse obstacles.",
        "urgent_table_header": "Complaints & Feedback Table",
        "no_complaints_success": "No complaints require attention.",
        "search_header": "🔍 Search Conversational Records",
        "search_input_label": "Enter keywords to search chat database (e.g. 'suara', 'materi', 'pemateri'):",
        "col_timestamp": "Timestamp",
        "col_username": "Username",
        "col_message": "Pesan",
        "col_sentiment": "Sentiment",
        "col_title": "Video Title",
        "system_status": "System Status"
    }
    sentiment_map = {
        "Positive": "Positive", 
        "Negative": "Negative", 
        "Neutral": "Neutral - Discussion",
        "Neutral - Discussion": "Neutral - Discussion",
        "Attendance / Greeting": "Attendance / Greeting"
    }
    color_map = {
        "Positive": "#98c379", 
        "Negative": "#e06c75", 
        "Neutral - Discussion": "#abb2bf",
        "Attendance / Greeting": "#e5c07b"
    }

# -----------------------------------------------------------------------------
# DYNAMIC TIMESTAMP HELPER
# -----------------------------------------------------------------------------
def get_last_updated_time():
    local_csv_path = "bpsdm_gemini_sentiment_results.csv"
    if os.path.exists(local_csv_path):
        mtime = os.path.getmtime(local_csv_path)
        dt = datetime.datetime.fromtimestamp(mtime, tz=ZoneInfo('Asia/Jakarta'))
        
        # Translate weekday if Bahasa Indonesia is selected
        if lang_choice == "Bahasa Indonesia":
            indonesian_days = {
                "Monday": "Senin", "Tuesday": "Selasa", "Wednesday": "Rabu",
                "Thursday": "Kamis", "Friday": "Jumat", "Saturday": "Sabtu", "Sunday": "Minggu"
            }
            day_en = dt.strftime("%A")
            day_id = indonesian_days.get(day_en, day_en)
            return dt.strftime(f"{day_id}, %d %B %Y, %H:%M WIB")
        return dt.strftime("%A, %d %B %Y, %H:%M WIB")
    
    # Fallback calculation if file info is missing
    now = datetime.datetime.now(ZoneInfo('Asia/Jakarta'))
    days_to_last_thursday = (now.weekday() - 3) % 7
    last_thursday = now - datetime.timedelta(days=days_to_last_thursday)
    if now.weekday() == 3 and now.hour < 13:
        last_thursday = last_thursday - datetime.timedelta(days=7)
    last_thursday_13 = last_thursday.replace(hour=13, minute=0, second=0, microsecond=0)
    
    if lang_choice == "Bahasa Indonesia":
        return last_thursday_13.strftime("Kamis, %d %B %Y, 13:00 WIB")
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
                if lang_choice == "Bahasa Indonesia":
                    st.sidebar.success("⚡ Data dimuat langsung dari Google Sheets Cloud!")
                else:
                    st.sidebar.success("⚡ Loaded live from Google Sheets Cloud!")
                return df
        except Exception as e:
            if lang_choice == "Bahasa Indonesia":
                st.sidebar.warning(f"Gagal menghubungkan ke Google Sheets ({e}). Menggunakan cadangan lokal...")
            else:
                st.sidebar.warning(f"Could not connect to Google Sheets ({e}). Using local fallback...")
            
    # Fallback to local CSV file
    if os.path.exists(local_csv_path):
        df = pd.read_csv(local_csv_path)
        if lang_choice == "Bahasa Indonesia":
            st.sidebar.info("📂 Menggunakan cadangan data lokal (Local CSV).")
        else:
            st.sidebar.info("📂 Running on offline cached backup (Local CSV).")
        return df
    else:
        if lang_choice == "Bahasa Indonesia":
            st.sidebar.error("❌ Sumber data tidak ditemukan. Jalankan pipeline ETL terlebih dahulu.")
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
        "Sentiment": "Neutral - Discussion",
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

def simplify_title(title):
    match = re.search(r'(ASN Belajar Seri \d+)\s*\|\s*(\d{4})', title, re.IGNORECASE)
    if match:
        return f"{match.group(1)} {match.group(2)}"
    if '|' in title:
        parts = title.split('|')
        series = parts[0].strip()
        year_match = re.search(r'\b\d{4}\b', parts[1])
        if year_match:
            return f"{series} {year_match.group(0)}"
        return series
    return title


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
st.sidebar.title(t["title"])

last_updated = get_last_updated_time()
st.sidebar.markdown(f"""
**{t["status_title"]}**
*   **{t["last_sync"]}:**
    `{last_updated}`
*   *{t["updated_schedule"]}*
""")

if not df.empty:
    # 1. Video Selector Filter
    titles = [t["all_videos"]] + list(df["Video Title"].unique())
    selected_title = st.sidebar.selectbox(t["select_webinar"], titles)
    
    # Apply filter based on selection
    if selected_title != t["all_videos"]:
        filtered_df = df[df["Video Title"] == selected_title]
    else:
        filtered_df = df

    # Keep a copy of unfiltered selection to calculate attendance metrics
    raw_unfiltered_df = filtered_df.copy()
    
    # Calculate attendance/greeting count
    attendance_count = raw_unfiltered_df[raw_unfiltered_df["Sentiment"] == "Attendance / Greeting"].shape[0]

    # 2. Filter Spam Toggle
    filter_spam = st.sidebar.checkbox(t["filter_spam_toggle"], value=True)
    if filter_spam:
        filtered_df = filtered_df[filtered_df["Sentiment"] != "Attendance / Greeting"]

    # Map sentiments to display language
    filtered_df = filtered_df.copy()
    filtered_df["Sentiment_Display"] = filtered_df["Sentiment"].map(sentiment_map)
    raw_unfiltered_df["Sentiment_Display"] = raw_unfiltered_df["Sentiment"].map(sentiment_map)

    # 3. Metadata Information Block
    st.sidebar.write("---")
    st.sidebar.markdown(f"### {t['metadata_header']}")
    
    # Get the 3 unique webinars from the active dataframe
    webinar_metadata = df[["Video Title", "Video ID"]].drop_duplicates().head(3)
    for index, row in webinar_metadata.iterrows():
        title_truncated = row["Video Title"][:50] + "..." if len(row["Video Title"]) > 50 else row["Video Title"]
        st.sidebar.markdown(f"""
        **🎬 {title_truncated}**
        *   **ID:** `{row['Video ID']}`
        *   **Link:** [{t['open_yt']}](https://youtube.com/watch?v={row['Video ID']})
        """)
        
    # -------------------------------------------------------------------------
    # MAIN DASHBOARD PAGE
    # -------------------------------------------------------------------------
    st.title(t["title"])
    st.subheader(t["subtitle"])
    st.markdown(f"*{t['last_updated_prefix']}{last_updated}***")
    
    # -------------------------------------------------------------------------
    # SECTION 1: KEY PERFORMANCE INDICATORS (KPIs)
    # -------------------------------------------------------------------------
    st.write(f"### {t['kpi_header']}")
    total_chats = len(filtered_df)
    
    # Sentiment calculations (uses active filtered dataset)
    counts = filtered_df["Sentiment"].value_counts()
    pos_count = counts.get("Positive", 0)
    neg_count = counts.get("Negative", 0)
    
    pos_pct = (pos_count / total_chats * 100) if total_chats > 0 else 0
    neg_pct = (neg_count / total_chats * 100) if total_chats > 0 else 0
    
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    with col_kpi1:
        st.metric(
            label=t["kpi1_label"], 
            value=f"{total_chats:,}",
            help=t["kpi1_help"]
        )
    with col_kpi2:
        st.metric(
            label=t["kpi2_label"], 
            value=f"{pos_pct:.1f}%",
            help=t["kpi2_help"]
        )
    with col_kpi3:
        st.metric(
            label=t["kpi3_label"], 
            value=f"{neg_pct:.1f}%",
            help=t["kpi3_help"]
        )
        
    st.write("---")
    
    # -------------------------------------------------------------------------
    # SECTION 2: SENTIMENT DISTRIBUTION AND TREND CHARTS
    # -------------------------------------------------------------------------
    st.write(f"### {t['charts_header']}")
    
    col_vis_left, col_vis_right = st.columns([1, 1])
    
    with col_vis_left:
        st.write(f"#### {t['donut_header']}")
        if total_chats > 0:
            fig_donut = px.pie(
                filtered_df,
                names="Sentiment_Display",
                color="Sentiment_Display",
                color_discrete_map=color_map,
                hole=0.45
            )
            fig_donut.update_traces(
                textposition='inside',
                textinfo='percent',
                textfont_size=24
            )
            fig_donut.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(size=13, color="#ffffff"),
                legend=dict(font=dict(size=13, color="#ffffff")),
                margin=dict(t=20, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_donut, use_container_width=True)
        else:
            st.info(t["no_data"])
            
    with col_vis_right:
        st.write(f"#### {t['bar_header']}")
        if total_chats > 0:
            # Group by Video Title and Sentiment, calculate percentage
            comp_df = filtered_df.groupby(["Video Title", "Sentiment_Display"]).size().reset_index(name="Count")
            comp_df["Video Title"] = comp_df["Video Title"].apply(simplify_title)
            comp_df["Percentage"] = comp_df.groupby("Video Title")["Count"].transform(lambda x: (x / x.sum()) * 100)
            
            fig_comp = px.bar(
                comp_df,
                x="Video Title",
                y="Percentage",
                color="Sentiment_Display",
                color_discrete_map=color_map,
                barmode="group",
                labels={"Video Title": t["bar_x"], "Percentage": t["bar_y"], "Sentiment_Display": t["col_sentiment"]}
            )
            fig_comp.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#ffffff"),
                xaxis=dict(showgrid=False, title=None, tickfont=dict(size=13)),
                yaxis=dict(showgrid=True, gridcolor="#222533", range=[0, 100]),
                legend=dict(font=dict(color="#ffffff")),
                margin=dict(t=20, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_comp, use_container_width=True)
        else:
            st.info(t["no_data"])
            
    # Timeline Trend Analysis
    st.write(f"#### {t['timeline_header']}")
    if total_chats > 0:
        timeline_df = filtered_df.copy()
        # Convert seconds to 5-minute interval blocks
        timeline_df["Minute"] = (timeline_df["Waktu"] // 300) * 5
        
        trend_df = timeline_df.groupby(["Minute", "Sentiment_Display"]).size().reset_index(name="Count")
        
        # Checkbox to isolate and view only complaints/negative trends
        isolate_complaints = st.checkbox(t["timeline_toggle"])
        
        neg_label = sentiment_map["Negative"]
        if isolate_complaints:
            trend_df = trend_df[trend_df["Sentiment_Display"] == neg_label]
            trend_color_map = {neg_label: "#e06c75"}
        else:
            trend_color_map = color_map
            
        fig_line = px.line(
            trend_df,
            x="Minute",
            y="Count",
            color="Sentiment_Display",
            color_discrete_map=trend_color_map,
            labels={"Minute": t["timeline_x"], "Count": t["timeline_y"], "Sentiment_Display": t["col_sentiment"]},
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
        st.info(t["no_data"])
        
    st.write("---")
    
    # -------------------------------------------------------------------------
    # SECTION 3: PPID CONTROL ROOM (COMPLAINT ANALYSIS)
    # -------------------------------------------------------------------------
    st.write(f"### {t['ppid_header']}")
    
    col_ctrl_left, col_ctrl_right = st.columns([1, 1])
    
    # Filter dataset for negative complaints
    negative_df = filtered_df[filtered_df["Sentiment"] == "Negative"].copy()
    
    with col_ctrl_left:
        st.write(f"#### {t['obstacles_header']}")
        if not negative_df.empty:
            neg_words = get_word_frequencies(negative_df["Pesan"])
            if neg_words:
                w_df = pd.DataFrame(neg_words, columns=["Keyword", "Frequency"])
                w_df = w_df.rename(columns={"Keyword": t["col_timestamp"], "Frequency": t["timeline_y"]})
                
                fig_neg_bar = px.bar(
                    w_df,
                    x=t["timeline_y"],
                    y=t["col_timestamp"],
                    orientation="h",
                    color=t["timeline_y"],
                    color_continuous_scale=px.colors.sequential.Reds
                )
                fig_neg_bar.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=False, tickfont=dict(color="#ffffff", size=13)),
                    yaxis=dict(showgrid=False, tickfont=dict(color="#ffffff", size=13), categoryorder="total ascending"),
                    coloraxis_showscale=False,
                    margin=dict(t=10, b=10, l=10, r=10)
                )
                st.plotly_chart(fig_neg_bar, use_container_width=True)
            else:
                st.info(t["not_enough_keywords"])
        else:
            st.success(t["no_negatives"])
            
    with col_ctrl_right:
        st.write(f"#### {t['urgent_table_header']}")
        if not negative_df.empty:
            # Format time offset to readable MM:SS
            negative_df["Timestamp"] = negative_df["Waktu"].apply(format_time_offset)
            
            # Display latest/most urgent 10 negative chats
            display_cols = ["Timestamp", "Username", "Pesan", "Video Title"]
            df_to_show = negative_df[display_cols].head(10)
            
            # Translate display columns headers
            df_to_show.columns = [t["col_timestamp"], t["col_username"], t["col_message"], t["col_title"]]
            
            st.dataframe(
                df_to_show,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.success(t["no_complaints_success"])
            
    st.write("---")
    
    # -------------------------------------------------------------------------
    # SECTION 4: CONVERSATIONAL SEARCH RECORDS
    # -------------------------------------------------------------------------
    st.write(f"### {t['search_header']}")
    search_query = st.text_input(t["search_input_label"])
    
    display_df = filtered_df.copy()
    if search_query:
        display_df = display_df[display_df["Pesan"].str.contains(search_query, case=False, na=False)]
        
    display_df["Timestamp"] = display_df["Waktu"].apply(format_time_offset)
    df_search_show = display_df[["Timestamp", "Username", "Pesan", "Sentiment_Display", "Video Title"]]
    
    # Translate search results table columns headers
    df_search_show.columns = [t["col_timestamp"], t["col_username"], t["col_message"], t["col_sentiment"], t["col_title"]]
    
    st.dataframe(
        df_search_show,
        use_container_width=True,
        hide_index=True
    )
    
else:
    st.info("Please verify data source configuration. No records loaded.")
