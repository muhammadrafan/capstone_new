"""
Module untuk integrasi dengan Ollama API untuk chatbot dan ringkasan
"""

import logging
from typing import Dict, Any, List, Optional
import random
import requests
import json

from helpers.config import OLLAMA_HOST, OLLAMA_MODEL, CHATBOT_TEMPLATE

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ollama_client")

def check_ollama_available() -> bool:
    """
    Periksa apakah server Ollama tersedia dan bisa diakses
    
    Returns:
        Boolean menandakan server tersedia atau tidak
    """
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags")
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama server tidak tersedia: {str(e)}")
        return False

def check_model_available(model_name: str = OLLAMA_MODEL) -> bool:
    """
    Periksa apakah model yang dibutuhkan sudah terinstal di Ollama
    
    Args:
        model_name: Nama model untuk diperiksa
        
    Returns:
        Boolean menandakan model tersedia atau tidak
    """
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags")
        if response.status_code == 200:
            models = response.json().get('models', [])
            for model_info in models:
                if model_info.get('name') == model_name:
                    return True
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Gagal memeriksa model: {str(e)}")
        return False

def pull_model(model_name: str = OLLAMA_MODEL) -> bool:
    """
    Download model jika belum tersedia
    
    Args:
        model_name: Nama model untuk di-download
        
    Returns:
        Boolean menandakan sukses atau gagal
    """
    try:
        # Pull model
        response = requests.post(
            f"{OLLAMA_HOST}/api/pull", 
            json={"name": model_name}
        )
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        logger.error(f"Gagal mengunduh model: {str(e)}")
        return False

def generate_conclusion(
    description: str, 
    sentiment_summary: str,
    model_name: str = OLLAMA_MODEL
) -> str:
    """
    Menghasilkan kesimpulan produk menggunakan model LLM
    
    Args:
        description: Deskripsi produk
        sentiment_summary: Ringkasan sentimen
        model_name: Nama model untuk digunakan
        
    Returns:
        String kesimpulan produk
    """
    try:
        # Buat prompt
        prompt = (
            "System: Kamu adalah asisten yang memberikan kesimpulan produk secara ringkas, objektif, dan alami berdasarkan data deskripsi dan sentimen.\n"
            "User: Buatkan kesimpulan apakah produk ini bagus dan worth it atau tidak, dengan gaya bahasa alami dan manusiawi. "
            "Gunakan informasi dari deskripsi dan ringkasan sentimen berikut.\n\n"
            f"Deskripsi produk:\n{description}\n\n"
            f"Ringkasan sentimen:\n{sentiment_summary}\n\n"
            "Berikan kesimpulan 3-5 kalimat, dengan bahasa Indonesia yang baik dan benar.\n\n"
            "Kesimpulan:"
        )
        
        # Generate dengan Ollama API
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 40
                }
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get('response', '').strip()
        else:
            logger.error(f"Gagal generate kesimpulan: {response.status_code}, {response.text}")
            return "Tidak dapat menghasilkan kesimpulan. Silakan periksa ulasan produk secara manual."
    
    except Exception as e:
        logger.error(f"Error saat generate kesimpulan: {str(e)}")
        return "Tidak dapat menghasilkan kesimpulan karena error sistem."

def get_chat_response(
    user_question: str,
    product_data: Dict[str, Any],
    model_name: str = OLLAMA_MODEL
) -> str:
    """
    Mendapatkan respons chatbot dari model untuk pertanyaan pengguna
    
    Args:
        user_question: Pertanyaan pengguna
        product_data: Data produk lengkap
        model_name: Nama model untuk digunakan
        
    Returns:
        String respons dari chatbot
    """
    try:
        # Ambil sampel ulasan (max 5 ulasan)
        sample_reviews = []
        for i, review in enumerate(product_data.get('reviews', [])[:5]):
            sample_reviews.append(f"{i+1}. {review.get('Nama')}: \"{review.get('Ulasan')}\" (Rating: {review.get('Rating')}/5, Sentimen: {review.get('Sentimen')})")
        
        # Format template prompt
        prompt = CHATBOT_TEMPLATE.format(
            product_name=product_data.get('product_name', 'Produk tidak diketahui'),
            description=product_data.get('description', 'Deskripsi tidak tersedia'),
            review_count=len(product_data.get('reviews', [])),
            positive_count=product_data.get('sentiment_counts', {}).get('positive', 0),
            neutral_count=product_data.get('sentiment_counts', {}).get('neutral', 0),
            negative_count=product_data.get('sentiment_counts', {}).get('negative', 0),
            conclusion=product_data.get('conclusion', 'Kesimpulan tidak tersedia'),
            sample_reviews="\n".join(sample_reviews),
            user_question=user_question
        )
        
        # Generate dengan Ollama API
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.8,
                    "top_p": 0.9,
                    "top_k": 40
                }
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get('response', '').strip()
        else:
            logger.error(f"Gagal generate chat response: {response.status_code}, {response.text}")
            return "Maaf, saya tidak dapat menjawab pertanyaan Anda saat ini. Silakan coba lagi nanti."
    
    except Exception as e:
        logger.error(f"Error saat generate chat response: {str(e)}")
        return "Maaf, terjadi kesalahan saat memproses pertanyaan Anda."

def setup_ollama() -> bool:
    """
    Setup Ollama dan pastikan model yang dibutuhkan tersedia
    
    Returns:
        Boolean menandakan sukses atau gagal
    """
    # Periksa apakah Ollama server tersedia
    if not check_ollama_available():
        logger.error("Ollama server tidak tersedia. Pastikan Ollama sudah diinstal dan berjalan.")
        return False
    
    # Periksa apakah model sudah tersedia
    if not check_model_available():
        logger.info(f"Model {OLLAMA_MODEL} belum tersedia, mencoba mengunduh...")
        if not pull_model():
            logger.error(f"Gagal mengunduh model {OLLAMA_MODEL}")
            return False
        logger.info(f"Model {OLLAMA_MODEL} berhasil diunduh")
    
    logger.info("Ollama setup selesai, siap digunakan")
    return True