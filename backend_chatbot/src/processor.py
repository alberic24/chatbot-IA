import os
import easyocr
import numpy as np
from pypdf import PdfReader
from pdf2image import convert_from_path


reader_ocr = easyocr.Reader(['fr', 'en'], gpu=False)

def extract_text_from_pdf(pdf_path):
    """Lit le texte d'un PDF et utilise l'OCR pour les pages scannées."""
    text = ""
    try:
        reader = PdfReader(pdf_path)
        classic_text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                classic_text += content + "\n"
        
        if len(classic_text.strip()) < 100:
            print("PDF semble scanné ou vide. Lancement de l'OCR page par page...")
            images = convert_from_path(pdf_path)
            ocr_text = ""
            for img in images:
                img_array = np.array(img)
                result = reader_ocr.readtext(img_array, detail=0)
                ocr_text += " ".join(result) + "\n"
            return ocr_text
        
        return classic_text

    except Exception as e:
        print(f"Erreur sur Fedora (poppler probable) : {e}")
        return ""

def extract_text_from_image(image_path):
    """Transforme une photo (JPG, PNG) en texte via EasyOCR."""
    try:
        result = reader_ocr.readtext(image_path, detail=0)
        return " ".join(result)
    except Exception as e:
        print(f"Erreur OCR Image : {e}")
        return ""

def process_file(file_path):
    """Détecte l'extension et choisit la bonne méthode."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext in [".jpg", ".jpeg", ".png"]:
        return extract_text_from_image(file_path)
    return None
