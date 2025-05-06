"""
Aplikasi utama QuickShop - aplikasi analisis sentimen produk Tokopedia dengan
integrasi Ollama untuk chatbot
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import base64
from io import BytesIO
import time
import os
import logging

# Import modul helper
from helpers.config import CUSTOM_CSS, MAX_REVIEWS_DEFAULT, BROWSER_HEADLESS_DEFAULT
from helpers.scraper import scrape_tokopedia_reviews, validate_tokopedia_url
from helpers.analyzer import (
    analyze_sentiment, generate_wordcloud, count_sentiments, 
    get_sentiment_summary, load_sentiment_model
)
from helpers.ollama_client import (
    setup_ollama, generate_conclusion, get_chat_response
)
from helpers.utils import (
    create_directories, save_product_data, format_product_name_for_filename
)

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("quickshop")

# Inisialisasi direktori
create_directories()

# Setup aplikasi Streamlit
st.set_page_config(
    page_title="QuickShop: Tokopedia Product Analysis",
    page_icon="🛍️",
    layout="wide"
)

# Apply custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def initialize_session_state():
    """
    Initialize Streamlit session state variables
    """
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "product_data" not in st.session_state:
        st.session_state.product_data = None
    
    if "ollama_available" not in st.session_state:
        # Cek ketersediaan Ollama
        st.session_state.ollama_available = setup_ollama()

def display_header():
    """
    Tampilkan header aplikasi
    """
    st.markdown("<h1 class='main-header'>🛍️ QuickShop: Evaluasi Produk Tokopedia dengan AI</h1>", unsafe_allow_html=True)
    st.markdown("*Analisis sentimen lokal dari ulasan produk Tokopedia dengan bantuan Ollama*")

def setup_sidebar():
    """
    Setup sidebar dengan opsi konfigurasi
    """
    with st.sidebar:
        st.header("📋 Tentang QuickShop")
        st.write("""
        QuickShop membantu Anda menganalisis ulasan produk Tokopedia menggunakan kecerdasan buatan secara lokal.
        
        **Cara Menggunakan:**
        1. Masukkan link produk Tokopedia
        2. Tunggu proses scraping dan analisis selesai
        3. Eksplor hasil analisis di tab yang tersedia
        4. Gunakan chatbot untuk mendapatkan informasi lebih lanjut
        """)
        
        st.markdown("---")
        
        # Chrome Configuration
        st.subheader("⚙️ Konfigurasi Chrome")
        headless_mode = st.checkbox("Mode Headless (tanpa GUI)", value=BROWSER_HEADLESS_DEFAULT,
                                   help="Aktifkan untuk menyembunyikan jendela Chrome saat scraping")
        
        max_reviews = st.number_input(
            "Jumlah ulasan maksimal", 
            min_value=10, 
            max_value=200, 
            value=MAX_REVIEWS_DEFAULT,
            help="Jumlah maksimal ulasan yang akan diambil"
        )
        
        # Ollama status
        st.markdown("---")
        st.subheader("🤖 Status Ollama")
        
        if st.session_state.ollama_available:
            st.success("✅ Ollama tersedia dan siap digunakan")
        else:
            st.error("""
            ❌ Ollama tidak tersedia
            
            Pastikan:
            1. Ollama sudah terinstal
            2. Server Ollama berjalan
            3. Model bahasa Indonesia sudah diunduh
            """)
            
            if st.button("🔄 Coba Lagi"):
                st.session_state.ollama_available = setup_ollama()
                st.experimental_rerun()
                
        st.markdown("---")
        st.write("QuickShop - All-in-One Tokopedia Product Analyzer")
        
        return headless_mode, max_reviews

def process_product_url(product_url, headless_mode, max_reviews):
    """
    Proses URL produk dan lakukan scraping
    
    Args:
        product_url: URL produk Tokopedia
        headless_mode: Boolean untuk mode headless
        max_reviews: Jumlah maksimum ulasan
        
    Returns:
        Data produk hasil scraping dan analisis
    """
    if not validate_tokopedia_url(product_url):
        st.warning("⚠️ Harap masukkan link produk Tokopedia yang valid!")
        return None
    
    # Create containers for each step
    scraping_container = st.container()
    
    with scraping_container:
        st.subheader("🔄 Proses Scraping dan Analisis")
        scraping_status = st.empty()
        scraping_progress = st.progress(0)
        
        # Setup callback function untuk status
        def update_status(message, is_progress=False):
            if is_progress:
                scraping_progress.progress(message)
            else:
                scraping_status.markdown(message, unsafe_allow_html=True)
        
        # Lakukan scraping
        update_status("⏳ Memulai proses scraping...")
        scraped_data = scrape_tokopedia_reviews(
            product_url, 
            max_reviews=max_reviews, 
            headless=headless_mode,
            status_callback=update_status
        )
        
        if not scraped_data:
            st.error("❌ Gagal melakukan scraping. Silakan periksa URL produk atau coba lagi nanti.")
            return None
        
        # Tampilkan info produk awal
        product_name = scraped_data['product_name']
        update_status(f"✅ Scraping selesai! Berhasil mendapatkan data produk: {product_name}")
        
        # Load sentiment model
        update_status("⏳ Memuat model analisis sentimen...")
        model_loaded = load_sentiment_model("quickshop-indobert-sentiment")
        if not model_loaded:
            st.warning("⚠️ Gagal memuat model sentimen. Menggunakan fallback.")
        
        # Analisis sentimen
        update_status("⏳ Menganalisis sentimen ulasan...")
        sentiments, preprocessed_texts, positive_counts, negative_counts = analyze_sentiment(scraped_data['reviews'])
        
        # Tambahkan hasil analisis ke data
        for i in range(len(scraped_data['reviews'])):
            scraped_data['reviews'][i]['Sentimen'] = sentiments[i]
            scraped_data['reviews'][i]['Preprocessed'] = preprocessed_texts[i]
            scraped_data['reviews'][i]['Positive_Count'] = positive_counts[i]
            scraped_data['reviews'][i]['Negative_Count'] = negative_counts[i]
        
        # Hitung statistik sentimen
        sentiment_counts = count_sentiments(sentiments)
        scraped_data['sentiment_counts'] = sentiment_counts
        
        # Generate sentiment summary
        sentiment_summary = get_sentiment_summary(sentiments)
        scraped_data['sentiment_summary'] = sentiment_summary
        
        # Generate wordcloud
        update_status("⏳ Membuat wordcloud...")
        wordcloud_base64 = generate_wordcloud(preprocessed_texts)
        scraped_data['wordcloud_base64'] = wordcloud_base64
        
        # Generate conclusion with Ollama
        if st.session_state.ollama_available:
            update_status("⏳ Menghasilkan kesimpulan dengan Ollama...")
            conclusion = generate_conclusion(
                scraped_data['description'],
                sentiment_summary
            )
            scraped_data['conclusion'] = conclusion
        else:
            # Fallback jika Ollama tidak tersedia
            scraped_data['conclusion'] = "Untuk mendapatkan kesimpulan produk otomatis, pastikan Ollama tersedia dan berjalan."
        
        # Selesai
        update_status("✅ Analisis selesai!")
        scraping_progress.progress(1.0)
        
        return scraped_data

def display_product_info(product_data):
    """
    Tampilkan informasi produk
    
    Args:
        product_data: Data produk untuk ditampilkan
    """
    st.markdown("<h3 class='sub-header'>🏷️ Informasi Produk</h3>", unsafe_allow_html=True)
    
    st.markdown("<div class='highlight'>", unsafe_allow_html=True)
    st.subheader(product_data['product_name'])
    st.markdown("### 📝 Deskripsi Produk")
    st.write(product_data['description'])
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Kesimpulan AI
    st.markdown("<h3>🔍 Kesimpulan AI</h3>", unsafe_allow_html=True)
    st.markdown(f"<div class='conclusion'>{product_data['conclusion']}</div>", unsafe_allow_html=True)

def display_sentiment_analysis(product_data):
    """
    Tampilkan analisis sentimen
    
    Args:
        product_data: Data produk untuk ditampilkan
    """
    st.markdown("<h3 class='sub-header'>📊 Statistik Ulasan</h3>", unsafe_allow_html=True)
    st.write(f"**Total Ulasan yang Dianalisis:** {len(product_data['reviews'])}")
    
    sentiment_counts = product_data['sentiment_counts']
    
    # Display sentiment counts with custom CSS
    col_pos, col_neu, col_neg = st.columns(3)
    with col_pos:
        st.markdown("<div class='metrics-container positive-metric'>", unsafe_allow_html=True)
        st.metric("Positif ✅", sentiment_counts['positive'], delta=None)
        st.markdown("</div>", unsafe_allow_html=True)
    with col_neu:
        st.markdown("<div class='metrics-container neutral-metric'>", unsafe_allow_html=True)
        st.metric("Netral 🟡", sentiment_counts['neutral'], delta=None)
        st.markdown("</div>", unsafe_allow_html=True)
    with col_neg:
        st.markdown("<div class='metrics-container negative-metric'>", unsafe_allow_html=True)
        st.metric("Negatif ❌", sentiment_counts['negative'], delta=None)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Calculate percentages
    total = sum(sentiment_counts.values())
    if total > 0:
        percentages = {
            'Positif': round(sentiment_counts['positive'] / total * 100, 1),
            'Netral': round(sentiment_counts['neutral'] / total * 100, 1),
            'Negatif': round(sentiment_counts['negative'] / total * 100, 1)
        }
        
        # Display sentiment percentages
        st.subheader("Persentase Sentimen")
        st.write(f"Positif: {percentages['Positif']}% | Netral: {percentages['Netral']}% | Negatif: {percentages['Negatif']}%")
    
    # Display sentiment chart
    st.subheader("Distribusi Sentimen")
    sentiment_data = {
        'Sentimen': ['Positif', 'Netral', 'Negatif'],
        'Jumlah': [sentiment_counts['positive'], sentiment_counts['neutral'], sentiment_counts['negative']]
    }
    
    sentiment_df = pd.DataFrame(sentiment_data)
    st.bar_chart(sentiment_df, x='Sentimen', y='Jumlah')
    
    # Display pie chart
    st.subheader("Proporsi Sentimen")
    fig1, ax1 = plt.subplots()
    ax1.pie(
        [sentiment_counts['positive'], sentiment_counts['neutral'], sentiment_counts['negative']], 
        labels=['Positif', 'Netral', 'Negatif'],
        autopct='%1.1f%%',
        startangle=90,
        colors=['#4CAF50', '#FFC107', '#F44336']
    )
    ax1.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle
    st.pyplot(fig1)

def display_wordcloud(product_data):
    """
    Tampilkan wordcloud
    
    Args:
        product_data: Data produk untuk ditampilkan
    """
    st.markdown("<h3 class='sub-header'>☁️ Word Cloud dari Ulasan</h3>", unsafe_allow_html=True)
    st.write("Visualisasi kata-kata yang sering muncul dalam ulasan produk:")
    
    # Convert base64 to image
    if product_data.get('wordcloud_base64'):
        try:
            wordcloud_img = Image.open(BytesIO(base64.b64decode(product_data['wordcloud_base64'])))
            st.image(wordcloud_img, caption='Word Cloud Ulasan Produk', use_column_width=True)
        except Exception as e:
            st.error(f"Gagal menampilkan wordcloud: {str(e)}")
    else:
        st.warning("Word cloud tidak tersedia")

def display_reviews(product_data):
    """
    Tampilkan detail ulasan
    
    Args:
        product_data: Data produk untuk ditampilkan
    """
    st.markdown("<h3 class='sub-header'>📋 Detail Ulasan</h3>", unsafe_allow_html=True)
    
    # Konversi ke DataFrame
    df = pd.DataFrame(product_data['reviews'])
    
    # Tampilkan informasi jumlah data
    st.write(f"Menampilkan {len(df)} ulasan produk")
    
    # Filter columns untuk ditampilkan
    display_columns = ['Nama', 'Rating', 'Ulasan', 'Sentimen']
    missing_columns = [col for col in display_columns if col not in df.columns]
    
    if missing_columns:
        st.warning(f"Beberapa kolom tidak ditemukan: {', '.join(missing_columns)}")
        st.dataframe(df)
    else:
        # Filter untuk sentiment
        sentiment_filter = st.multiselect(
            "Filter berdasarkan sentimen:", 
            options=["Positif", "Netral", "Negatif"],
            default=["Positif", "Netral", "Negatif"]
        )
        
        # Filter DataFrame
        if sentiment_filter:
            filtered_df = df[df['Sentimen'].isin(sentiment_filter)]
        else:
            filtered_df = df
        
        # Tampilkan DataFrame
        st.dataframe(
            filtered_df[display_columns],
            column_config={
                "Nama": st.column_config.TextColumn("Nama Pengguna"),
                "Rating": st.column_config.NumberColumn("Rating", format="%d ⭐"),
                "Ulasan": st.column_config.TextColumn("Ulasan"),
                "Sentimen": st.column_config.TextColumn(
                    "Sentimen",
                    help="Hasil analisis sentimen",
                    width="medium"
                )
            },
            use_container_width=True
        )

def display_chatbot(product_data):
    """
    Tampilkan antarmuka chatbot
    
    Args:
        product_data: Data produk untuk konteks chatbot
    """
    st.markdown("<h3 class='sub-header'>🤖 Tanya AI tentang Produk</h3>", unsafe_allow_html=True)
    
    if not st.session_state.ollama_available:
        st.warning(
            "⚠️ Chatbot membutuhkan Ollama untuk berfungsi. "
            "Pastikan Ollama tersedia dan model bahasa Indonesia sudah diunduh."
        )
        return
    
    # Tampilkan riwayat chat
    chat_container = st.container()
    with chat_container:
        for i, message in enumerate(st.session_state.chat_history):
            if message["role"] == "user":
                st.markdown(f"<div class='user-message'><b>Anda:</b> {message['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='bot-message'><b>AI:</b> {message['content']}</div>", unsafe_allow_html=True)
    
    # Input untuk pertanyaan baru
    user_question = st.text_input("💬 Tanyakan tentang produk ini:", placeholder="Contoh: Apakah produk ini worth it?")
    
    # Tombol untuk mengirim pertanyaan
    if st.button("Kirim Pertanyaan"):
        if user_question:
            # Tambahkan pertanyaan user ke history
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            
            # Tampilkan loading spinner
            with st.spinner("AI sedang berpikir..."):
                # Dapatkan respons dari Ollama
                bot_response = get_chat_response(user_question, product_data)
                
                # Tambahkan respons bot ke history
                st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
            
            # Refresh tampilan untuk memperbarui riwayat chat
            st.rerun()

def main():
    """
    Fungsi utama aplikasi
    """
    # Inisialisasi session state
    initialize_session_state()
    
    # Tampilkan header
    display_header()
    
    # Setup sidebar dan dapatkan konfigurasi
    headless_mode, max_reviews = setup_sidebar()
    
    # Main content
    st.markdown("<h2 class='sub-header'>🔍 Analisis Produk</h2>", unsafe_allow_html=True)

    # Input URL
    product_url = st.text_input("🔗 Masukkan link produk Tokopedia:", placeholder="https://www.tokopedia.com/store/product-name")

    # Process button
    if st.button("🚀 Analisis Sentimen"):
        if product_url:
            # Proses URL produk
            product_data = process_product_url(product_url, headless_mode, max_reviews)
            
            if product_data:
                # Simpan data produk ke session state
                st.session_state.product_data = product_data
                
                # Reset chat history saat menganalisis produk baru
                st.session_state.chat_history = []
                
                # Simpan data produk ke file
                product_filename = format_product_name_for_filename(product_data['product_name'])
                save_product_data(product_data, product_filename)
                
                # Tampilkan hasil menggunakan tabs
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "Informasi Produk", 
                    "Analisis Sentimen", 
                    "Word Cloud", 
                    "Detail Ulasan",
                    "Chatbot"
                ])
                
                with tab1:
                    display_product_info(product_data)
                
                with tab2:
                    display_sentiment_analysis(product_data)
                
                with tab3:
                    display_wordcloud(product_data)
                
                with tab4:
                    display_reviews(product_data)
                
                with tab5:
                    display_chatbot(product_data)
    
    # Display existing product data if available
    elif st.session_state.product_data:
        # Tampilkan hasil menggunakan tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Informasi Produk", 
            "Analisis Sentimen", 
            "Word Cloud", 
            "Detail Ulasan",
            "Chatbot"
        ])
        
        with tab1:
            display_product_info(st.session_state.product_data)
        
        with tab2:
            display_sentiment_analysis(st.session_state.product_data)
        
        with tab3:
            display_wordcloud(st.session_state.product_data)
        
        with tab4:
            display_reviews(st.session_state.product_data)
        
        with tab5:
            display_chatbot(st.session_state.product_data)
    
    else:
        # Show help text when starting
        st.info("""
        **Cara Menggunakan QuickShop:**
        1. Masukkan link produk Tokopedia di kolom input di atas
        2. Anda dapat memilih untuk menampilkan jendela Chrome dengan menonaktifkan "Mode Headless" di sidebar
        3. Klik tombol "Analisis Sentimen"
        4. Tunggu proses scraping dan analisis selesai
        5. Eksplor hasil analisis di tab yang tersedia
        6. Gunakan chatbot untuk mendapatkan informasi lebih lanjut
        
        **Catatan:** 
        - Proses scraping akan membuka jendela Chrome jika mode headless dinonaktifkan
        - Semua proses analisis dilakukan secara lokal di komputer Anda
        - Untuk menggunakan chatbot, pastikan Ollama tersedia dan model bahasa Indonesia sudah diunduh
        """)
        
        # Placeholder for result layout
        st.markdown("<h2 class='sub-header'>🏷️ Informasi Produk</h2>", unsafe_allow_html=True)
        st.write("Hasil analisis akan muncul di sini setelah Anda memasukkan link produk.")

if __name__ == "__main__":
    main()