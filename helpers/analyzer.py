"""
Module untuk analisis sentimen dan pengolahan ulasan
"""

import re
import base64
from io import BytesIO
import logging
from typing import List, Dict, Tuple, Any, Optional
import numpy as np

import matplotlib.pyplot as plt
from wordcloud import WordCloud
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from helpers.config import SENTIMENT_LABELS, POSITIVE_WORDS, NEGATIVE_WORDS

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("analyzer")

# Variable global untuk model dan tokenizer
tokenizer = None
model = None
models_loaded = False

def replace_emojis(text: str) -> str:
    """
    Mengganti emoji dengan teks yang sesuai
    
    Args:
        text: Teks yang mengandung emoji
        
    Returns:
        Teks yang sudah diganti emojinya
    """
    emoji_dict = {
        "😍": "senang", "❤️": "cinta", "💔": "sedih", "😡": "marah", "😢": "sedih",
        "😊": "senang", "😁": "senang", "😭": "sedih", "👍": "bagus", "👎": "jelek",
        "🥰": "senang", "💖": "cinta", "💗": "cinta", "💕": "cinta", "💞": "cinta", 
        "😞": "kecewa", "😔": "kecewa", "😃": "senang", "🤗": "senang", "😎": "keren"
    }
    for emoji, replacement in emoji_dict.items():
        text = text.replace(emoji, f" {replacement} ")
    return text

def preprocess_text(text: str) -> str:
    """
    Praproses teks ulasan sebelum analisis sentimen
    
    Args:
        text: Teks ulasan asli
        
    Returns:
        Teks yang sudah diproses
    """
    text = str(text).lower()
    text = replace_emojis(text)
    
    # Penggantian singkatan dan kata sehari-hari
    text = text.replace("gk", "tidak").replace("tdk", "tidak").replace("ga", "tidak")
    text = text.replace("bgt", "banget").replace("bgtt", "banget").replace("bgttt", "banget")
    text = text.replace("ok", "oke")
    
    # Hapus karakter berulang
    text = re.sub(r'(.)\\1{2,}', r'\\1', text)
    
    # Hapus tanda baca
    text = re.sub(r'[^\\w\\s]', ' ', text)
    
    # Hapus spasi berlebih
    text = re.sub(r'\\s+', ' ', text).strip()
    
    # Coba menghapus stopword jika Sastrawi tersedia
    try:
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
        factory = StopWordRemoverFactory()
        stopword_remover = factory.create_stop_word_remover()
        text = stopword_remover.remove(text)
    except Exception as e:
        logger.warning(f"Couldn't remove stopwords: {str(e)}")
    
    # Hapus spasi berlebih lagi setelah penghapusan stopword
    text = re.sub(r'\\s+', ' ', text).strip()
    
    return text

def load_sentiment_model(model_path: Optional[str] = None) -> bool:
    """
    Muat model analisis sentimen
    
    Args:
        model_path: Path ke model sentiment analysis (opsional)
        
    Returns:
        Boolean sukses atau gagal
    """
    global tokenizer, model, models_loaded
    
    if models_loaded:
        return True
    
    try:
        logger.info("Loading sentiment analysis model...")
        
        # Opsi 1: Jika path model disediakan, gunakan model lokal
        if model_path:
            logger.info(f"Loading local model from {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            
        # Opsi 2: Gunakan model IndoBERT dari Hugging Face
        else:
            logger.info("Loading IndoBERT from Hugging Face")
            model_name = "indobenchmark/indobert-base-p1"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
        
        models_loaded = True
        logger.info("Sentiment model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error loading sentiment model: {str(e)}")
        return False

def analyze_sentiment(reviews: List[Dict[str, Any]]) -> Tuple[List[str], List[str], List[int], List[int]]:
    """
    Analisis sentimen untuk daftar ulasan
    
    Args:
        reviews: List dari dictionary ulasan
        
    Returns:
        Tuple dari (label sentimen, teks yang sudah diproses, jumlah kata positif, jumlah kata negatif)
    """
    # Load model jika belum dimuat
    if not models_loaded:
        load_sentiment_model()
        
    sentiments = []
    preprocessed_texts = []
    positive_counts = []
    negative_counts = []
    
    for review in reviews:
        clean_text = preprocess_text(review["Ulasan"])
        preprocessed_texts.append(clean_text)

        # Hitung kata positif dan negatif
        positive_count = sum(1 for word in POSITIVE_WORDS if word in clean_text.lower())
        negative_count = sum(1 for word in NEGATIVE_WORDS if word in clean_text.lower())
        positive_counts.append(positive_count)
        negative_counts.append(negative_count)

        # Analisis sentimen menggunakan model
        inputs = tokenizer(clean_text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits.squeeze()
        sentiment = torch.argmax(logits).item()
        confidence = torch.softmax(logits, dim=-1)
        max_confidence = confidence[sentiment].item()

        # Koreksi sentimen berdasarkan rating, kata positif/negatif
        rating = review["Rating"]
        if sentiment == 0 and (rating == 3):
            sentiment = 1  # Ubah negatif ke netral jika rating 3
        elif sentiment == 2 and (rating == 3):
            sentiment = 1  # Ubah positif ke netral jika rating 3
        elif sentiment == 0 and (rating in [4, 5] or positive_count > negative_count):
            sentiment = 1  # Ubah negatif ke netral jika rating tinggi atau banyak kata positif
        elif sentiment == 2 and (rating in [1, 2] or negative_count > positive_count):
            sentiment = 1  # Ubah positif ke netral jika rating rendah atau banyak kata negatif
        elif sentiment == 1 and (rating in [4, 5] or positive_count > negative_count):
            sentiment = 2  # Ubah netral ke positif jika rating tinggi atau banyak kata positif
        elif sentiment == 1 and (rating in [1, 2] or negative_count > positive_count):
            sentiment = 0  # Ubah netral ke negatif jika rating rendah atau banyak kata negatif

        sentiments.append(SENTIMENT_LABELS[sentiment])
        
    return sentiments, preprocessed_texts, positive_counts, negative_counts

def generate_wordcloud(texts: List[str]) -> str:
    """
    Menghasilkan wordcloud dari kumpulan teks
    
    Args:
        texts: List teks ulasan
        
    Returns:
        String base64 dari gambar wordcloud
    """
    try:
        text = ' '.join(texts)
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=200,
            collocations=False,
            regexp=r'\w+'
        ).generate(text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")

        # Save to BytesIO object
        img_data = BytesIO()
        plt.savefig(img_data, format='png', bbox_inches='tight')
        img_data.seek(0)

        # Convert to base64 for easy transfer
        encoded = base64.b64encode(img_data.read()).decode('utf-8')
        plt.close()
        
        return encoded
    except Exception as e:
        logger.error(f"Error generating wordcloud: {str(e)}")
        # Return empty string on error
        return ""

def count_sentiments(sentiments: List[str]) -> Dict[str, int]:
    """
    Menghitung jumlah untuk setiap kategori sentimen
    
    Args:
        sentiments: List label sentimen
        
    Returns:
        Dictionary jumlah untuk setiap kategori
    """
    return {
        'positive': sentiments.count('Positif'),
        'neutral': sentiments.count('Netral'),
        'negative': sentiments.count('Negatif')
    }

def get_sentiment_summary(sentiments: List[str]) -> str:
    """
    Menghasilkan ringkasan sentimen dalam format teks
    
    Args:
        sentiments: List label sentimen
        
    Returns:
        String ringkasan sentimen
    """
    counts = count_sentiments(sentiments)
    
    return f"""
    ✅ **{counts['positive']}** ulasan positif
    🟡 **{counts['neutral']}** ulasan netral
    ❌ **{counts['negative']}** ulasan negatif
    """