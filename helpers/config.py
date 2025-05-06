"""
Konfigurasi aplikasi QuickShop
"""

# Ollama configuration
OLLAMA_HOST = "http://localhost:11434"  # URL default Ollama
OLLAMA_MODEL = "bangundwir/bahasa-4b-chat"  # Model Bahasa Indonesia untuk Ollama

# Scraper configuration
MAX_REVIEWS_DEFAULT = 50
BROWSER_HEADLESS_DEFAULT = True
TOKOPEDIA_DOMAIN = "tokopedia.com"

# Sentiment analysis configuration
SENTIMENT_LABELS = {
    0: "Negatif",
    1: "Netral",
    2: "Positif"
}

# Word list untuk analisis sentimen
POSITIVE_WORDS = [
    "mantap", "bagus", "jernih", "nyaman", "original", "premium", 
    "cepat", "murah", "berkualitas", "aman", "good", "good quality", 
    "terimakasih", "terima kasih", "memuaskan", "puas", "recommended",
    "worth it", "cocok", "enak", "awet", "tahan lama", "kuat"
]

NEGATIVE_WORDS = [
    "buruk", "jelek", "rusak", "lemot", "cacat", "lambat", "kurang", 
    "tidak bagus", "gagal", "kecewa", "mahal", "kemahalan", "kasar", 
    "bocor", "palsu", "pecah", "suram", "rugi", "tidak worth it"
]

# Konfigurasi style untuk Streamlit
CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF5722;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #FF5722;
        border-bottom: 2px solid #FF5722;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    .highlight {
        background-color: #FFF3E0;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .conclusion {
        background-color: #E3F2FD;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        color: #0D47A1;
        font-size: 1.1rem;
        font-weight: 500;
        line-height: 1.5;
    }
    .stProgress > div > div > div > div {
        background-color: #FF5722;
    }
    .scraping-container {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
        background-color: #f9f9f9;
    }
    .metrics-container {
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 15px;
    }
    .positive-metric {
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
    }
    .neutral-metric {
        background-color: #FFF8E1;
        border-left: 5px solid #FFC107;
    }
    .negative-metric {
        background-color: #FFEBEE;
        border-left: 5px solid #F44336;
    }
    .chat-container {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        margin-top: 20px;
        background-color: #f8f9fa;
    }
    .user-message {
        background-color: #e3f2fd;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .bot-message {
        background-color: #f1f1f1;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
</style>
"""

# Pesan template untuk chatbot
CHATBOT_TEMPLATE = """
Kamu adalah asisten AI untuk aplikasi QuickShop yang membantu pengguna mendapatkan informasi dan rekomendasi produk.
Kamu akan menjawab pertanyaan tentang produk: {product_name}.

Berikut adalah informasi yang kamu punya tentang produk:
1. Deskripsi produk: {description}
2. Jumlah ulasan: {review_count}
3. Sentimen: {positive_count} positif, {neutral_count} netral, {negative_count} negatif
4. Kesimpulan: {conclusion}

Berikut beberapa ulasan dari pengguna:
{sample_reviews}

Jawab pertanyaan pengguna dengan sopan, ringkas, dan berikan rekomendasi yang tepat berdasarkan informasi di atas.
Pertanyaan: {user_question}
"""