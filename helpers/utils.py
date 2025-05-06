"""
Modul utilitas untuk aplikasi QuickShop
"""

import os
import logging
from typing import Dict, Any, List
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("utils")

def create_directories():
    """
    Membuat direktori yang diperlukan untuk aplikasi
    """
    # Buat direktori untuk data jika belum ada
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    logger.info("Direktori aplikasi disiapkan")

def convert_to_dataframe(reviews: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Mengubah list dictionary menjadi pandas DataFrame
    
    Args:
        reviews: List dictionary ulasan
        
    Returns:
        pandas DataFrame
    """
    try:
        df = pd.DataFrame(reviews)
        return df
    except Exception as e:
        logger.error(f"Gagal mengubah ke DataFrame: {str(e)}")
        return pd.DataFrame()  # Return DataFrame kosong jika error

def save_product_data(product_data: Dict[str, Any], filename: str) -> bool:
    """
    Menyimpan data produk ke file CSV
    
    Args:
        product_data: Dictionary data produk
        filename: Nama file untuk menyimpan
        
    Returns:
        Boolean menandakan sukses atau gagal
    """
    try:
        # Konversi ke DataFrame
        df = convert_to_dataframe(product_data.get('reviews', []))
        
        # Simpan ke CSV
        filepath = os.path.join('data', f"{filename}.csv")
        df.to_csv(filepath, index=False)
        logger.info(f"Data produk disimpan ke {filepath}")
        return True
    except Exception as e:
        logger.error(f"Gagal menyimpan data produk: {str(e)}")
        return False

def load_product_data(filename: str) -> Dict[str, Any]:
    """
    Memuat data produk dari file CSV
    
    Args:
        filename: Nama file untuk dimuat
        
    Returns:
        Dictionary data produk
    """
    try:
        filepath = os.path.join('data', f"{filename}.csv")
        
        # Periksa apakah file ada
        if not os.path.exists(filepath):
            logger.error(f"File tidak ditemukan: {filepath}")
            return {}
        
        # Baca dari CSV
        df = pd.read_csv(filepath)
        
        # Konversi kembali ke format data produk
        product_data = {
            'reviews': df.to_dict('records')
        }
        
        logger.info(f"Data produk dimuat dari {filepath}")
        return product_data
    except Exception as e:
        logger.error(f"Gagal memuat data produk: {str(e)}")
        return {}

def format_product_name_for_filename(product_name: str) -> str:
    """
    Format nama produk agar aman digunakan sebagai nama file
    
    Args:
        product_name: Nama produk asli
        
    Returns:
        String nama file yang aman
    """
    # Hapus karakter yang tidak valid pada nama file
    invalid_chars = r'<>:"/\|?*'
    filename = ''.join(c for c in product_name if c not in invalid_chars)
    
    # Truncate jika terlalu panjang
    if len(filename) > 50:
        filename = filename[:50]
    
    # Replace spaces with underscores
    filename = filename.strip().replace(' ', '_')
    
    return filename