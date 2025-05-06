"""
Module untuk scraping data produk dan ulasan dari Tokopedia
"""

import time
import re
from typing import Dict, List, Optional, Any
import logging

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

from helpers.config import TOKOPEDIA_DOMAIN, MAX_REVIEWS_DEFAULT, BROWSER_HEADLESS_DEFAULT

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("tokopedia_scraper")

def setup_driver(headless: bool = True) -> webdriver.Chrome:
    """
    Setup Chrome driver untuk scraping
    
    Args:
        headless: Boolean untuk menjalankan browser tanpa GUI
        
    Returns:
        Objek webdriver.Chrome yang sudah dikonfigurasi
    """
    # Setup Chrome options
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless")
    
    chrome_options.add_argument("--start-maximized")  # Maximize window
    
    # Tambahkan opsi lain untuk mengurangi deteksi otomatisasi
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)
    
    # Create Chrome driver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    # Mengubah user agent untuk menghindari deteksi otomatisasi
    driver.execute_cdp_cmd('Network.setUserAgentOverride', {
        "userAgent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })
    
    return driver

def validate_tokopedia_url(url: str) -> bool:
    """
    Memvalidasi URL produk Tokopedia
    
    Args:
        url: URL produk untuk divalidasi
        
    Returns:
        Boolean menandakan URL valid atau tidak
    """
    return TOKOPEDIA_DOMAIN in url.lower() and "http" in url.lower()

def scrape_tokopedia_reviews(
    product_url: str, 
    max_reviews: int = MAX_REVIEWS_DEFAULT, 
    headless: bool = BROWSER_HEADLESS_DEFAULT,
    status_callback = None
) -> Optional[Dict[str, Any]]:
    """
    Scrape data produk dan ulasan dari Tokopedia
    
    Args:
        product_url: URL produk Tokopedia
        max_reviews: Jumlah maksimum ulasan yang akan diambil
        headless: Boolean untuk menjalankan browser tanpa GUI
        status_callback: Callback function untuk melaporkan status (opsional)
        
    Returns:
        Dictionary berisi data produk dan ulasan, atau None jika gagal
    """
    # Validasi URL
    if not validate_tokopedia_url(product_url):
        if status_callback:
            status_callback("❌ URL produk tidak valid! Pastikan ini adalah URL produk Tokopedia.")
        logger.error(f"URL tidak valid: {product_url}")
        return None
    
    # Setup status updates
    def update_status(message):
        logger.info(message)
        if status_callback:
            status_callback(f"<div class='scraping-container'>{message}</div>")
    
    update_status("⏳ Menyiapkan Chrome driver...")
    
    # Setup driver
    driver = setup_driver(headless=headless)
    
    try:
        # Arahkan ke URL produk
        update_status("⏳ Membuka halaman produk Tokopedia...")
        driver.get(product_url)
        time.sleep(7)  # Tunggu halaman dimuat
        
        # Tutup iklan jika ada
        try:
            update_status("⏳ Menangani popup...")
            div_iklan = driver.find_element(By.CLASS_NAME, "css-11hzwo5")
            iklan_button = div_iklan.find_element(By.TAG_NAME, "button")
            iklan_button.click()
            time.sleep(4)
        except Exception as e:
            update_status(f"ℹ️ Tidak ada popup untuk ditutup atau tidak dapat ditutup: {str(e)}")
        
        # Scroll ke bawah untuk memuat konten
        update_status("⏳ Memuat konten halaman...")
        driver.execute_script("window.scrollBy(0, 2000);")
        time.sleep(5)

        # Coba klik tombol "Lihat Selengkapnya" untuk deskripsi
        try:
            update_status("⏳ Mencoba membuka deskripsi lengkap...")
            see_more_button = driver.find_element(By.XPATH, "//button[@data-testid='btnPDPSeeMore']")
            see_more_button.click()
            time.sleep(5)
        except Exception as e:
            update_status(f"ℹ️ Tidak dapat membuka deskripsi lengkap: {str(e)}")
        
        # Ambil deskripsi dan nama produk
        update_status("⏳ Mengambil informasi produk...")
        soup = BeautifulSoup(driver.page_source, "html.parser")
        description_elem = soup.select_one("div[data-testid='lblPDPDescriptionProduk']")
        description = description_elem.get_text(strip=True) if description_elem else "Deskripsi tidak ditemukan"
        
        product_name_elem = soup.select_one("h1[data-testid='lblPDPDetailProductName']")
        product_name = product_name_elem.get_text(strip=True) if product_name_elem else "Produk Tidak Diketahui"

        update_status(f"✅ Produk terdeteksi: {product_name}")
        
        # Ambil ulasan
        reviews_data = []
        collected_reviews = set()
        
        # Hitung total ulasan
        try:
            total_elem = soup.find("p", {"data-testid": "reviewSortingSubtitle"})
            if total_elem:
                total_text = total_elem.get_text()
                match = re.search(r'dari (\d+)', total_text)
                if match:
                    total_available = int(match.group(1))
                    max_reviews = min(max_reviews, total_available)
                    update_status(f"ℹ️ Menemukan {total_available} ulasan, akan mengambil hingga {max_reviews}")
        except Exception as e:
            update_status(f"⚠️ Tidak dapat menghitung total ulasan: {str(e)}")
        
        # Proses halaman-halaman ulasan
        page = 1
        while len(reviews_data) < max_reviews:
            update_status(f"⏳ Memproses halaman ulasan {page}...")
            
            soup = BeautifulSoup(driver.page_source, "html.parser")
            containers = soup.find_all("article", class_="css-15m2bcr")
            
            if not containers:
                update_status("⚠️ Tidak ditemukan kontainer ulasan")
                break
                
            for container in containers:
                if len(reviews_data) >= max_reviews:
                    break
                    
                try:
                    review_elem = container.select_one("p span[data-testid='lblItemUlasan']")
                    review_text = review_elem.text.strip() if review_elem else "Tidak ada ulasan"
                    
                    if review_text in collected_reviews:
                        continue
                        
                    name_elem = container.select_one("div.css-k4rf3m span.name")
                    name = name_elem.text.strip() if name_elem else "Unknown"
                    
                    rating_elem = container.select_one("div[data-testid='icnStarRating']")
                    rating = rating_elem["aria-label"] if rating_elem else "Tidak ada rating"
                    rating = int(re.search(r'\d+', rating).group()) if rating != "Tidak ada rating" else 0
                    
                    reviews_data.append({"Nama": name, "Rating": rating, "Ulasan": review_text})
                    collected_reviews.add(review_text)
                    
                    # Report progress - percent completion
                    progress_percentage = min(len(reviews_data) / max_reviews, 1.0)
                    if status_callback:
                        status_callback(progress_percentage, is_progress=True)
                    
                except Exception as e:
                    update_status(f"⚠️ Error saat ekstraksi ulasan: {str(e)}")
            
            # Klik halaman berikutnya jika diperlukan
            if len(reviews_data) < max_reviews:
                try:
                    next_page_button = driver.find_element(By.XPATH, "//button[@aria-label='Laman berikutnya']")
                    next_page_button.click()
                    time.sleep(5)
                    page += 1
                except Exception as e:
                    update_status(f"⚠️ Tidak dapat beralih ke halaman berikutnya: {str(e)}")
                    break
        
        # Scraping selesai
        update_status(f"✅ Scraping selesai! Berhasil mengambil {len(reviews_data)} ulasan")
        
        # Siapkan data hasil scraping
        scraped_data = {
            "product_name": product_name,
            "description": description,
            "reviews": reviews_data
        }
        
        return scraped_data
        
    except Exception as e:
        update_status(f"❌ Error saat scraping: {str(e)}")
        logger.error(f"Error scraping: {str(e)}", exc_info=True)
        return None
    finally:
        # Tutup browser
        update_status("🔄 Menutup browser Chrome...")
        driver.quit()