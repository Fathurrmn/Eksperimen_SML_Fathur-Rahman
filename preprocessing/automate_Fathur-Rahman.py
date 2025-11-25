import pandas as pd
import numpy as np
import re
import os
import sys
import io 
from pathlib import Path
from sklearn.model_selection import train_test_split
import mlflow 

# --- PERBAIKAN: Set encoding output konsol ke UTF-8 untuk menghindari error emoji/non-ASCII ---
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')
# ------------------------------------------------------------------------------------------

# --- DAFTAR STOPWORDS MANUAL ---
ENGLISH_STOP_WORDS = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
])


def preprocess_text(text: str) -> str:
    """Fungsi pembersihan teks inti (Tanpa Lemmatization)."""
    if pd.isna(text) or text is None:
        return ""
        
    text = str(text).lower()

    # Hapus karakter non-alfabet (ganti dengan spasi)
    text = re.sub(r'[^a-z\s]', ' ', text)

    # Tokenisasi (dengan split sederhana)
    words = text.split()

    # Hapus stopwords dan word pendek
    cleaned = [w for w in words if w not in ENGLISH_STOP_WORDS and len(w) > 1]

    return " ".join(cleaned)


def sanitize_csv_for_windows(input_path: Path):
    """Membaca file sebagai byte dan membersihkan karakter non-ASCII (emoji, dll)."""
    print("Memproses data mentah untuk menghapus karakter non-ASCII yang menyebabkan crash...")
    try:
        # Baca file sebagai byte
        with open(input_path, 'rb') as f:
            content = f.read()
            
        # Decode menggunakan LATIN1, lalu encode kembali dengan errors='ignore' 
        sanitized_content = content.decode('latin1', errors='ignore').encode('utf-8', errors='ignore')
        return pd.read_csv(io.BytesIO(sanitized_content), encoding='utf-8')
    
    except Exception as e:
        raise Exception(f"Gagal total memuat dan membersihkan file CSV: {e}")


def automate_preprocessing_pipeline(
    input_file_path: str | Path,
    output_dir: str | Path = 'amazon_preprocessing'
) -> pd.DataFrame:
    """
    Melakukan seluruh tahapan preprocessing secara otomatis dan mencatat hasil ke MLflow.
    """
    input_file_path = Path(input_file_path)
    print(f"Memulai preprocessing untuk file: {input_file_path.name}")
    
    # SETUP MLFLOW
    mlflow.set_tracking_uri("http://127.0.0.1:5000/") 
    mlflow.set_experiment("Proyek Akhir Data Preprocessing") 
    
    with mlflow.start_run(run_name=f"Preprocessing_{input_file_path.name}"):
        
        # --- 1. Load Data ---
        if not input_file_path.exists():
            mlflow.log_param("data_error", f"Raw file not found: {input_file_path.resolve()}")
            raise FileNotFoundError(f"File not found: {input_file_path}")
        
        df = sanitize_csv_for_windows(input_file_path)
        
        mlflow.log_param("raw_data_path", str(input_file_path.resolve()))

        # --- 2. Kolom Teks dan Target Handling ---
        possible_text = ['review_text', 'review_content', 'review', 'about_product', 'review_title']
        text_col = next((c for c in possible_text if c in df.columns), None)
        if text_col is None:
            raise KeyError(f"No valid text column found. Available columns: {df.columns.tolist()}")

        # Menghasilkan kolom 'sentiment' jika hanya ada 'rating'
        if 'sentiment' not in df.columns and 'rating' in df.columns:
            df['rating_num'] = pd.to_numeric(df['rating'].astype(str).str.replace(',', '', regex=False), errors='coerce')
            df['sentiment_encoded'] = df['rating_num'].apply(lambda r: 1 if pd.notna(r) and r >= 4 else (0 if pd.notna(r) and r < 4 else np.nan))
            df['sentiment'] = df['sentiment_encoded'].map({1: 'positive', 0: 'negative'})
        elif 'sentiment' in df.columns and 'sentiment_encoded' not in df.columns:
            pass
        else:
            raise KeyError("No 'sentiment' or 'rating' column found to create a label.")

        if text_col != 'review':
            df.rename(columns={text_col: 'review'}, inplace=True)
            
        # --- 3. Data Cleaning (Duplikasi, Missing Value) ---
        initial_rows = len(df)
        df.drop_duplicates(inplace=True)
        
        df.dropna(subset=['review', 'sentiment'], inplace=True)
        
        df['review'] = df['review'].fillna("").apply(preprocess_text)
        
        # --- 4. Encoding dan Pembersihan Target (lanjutan) ---
        if 'sentiment_encoded' not in df.columns:
            df['sentiment'] = df['sentiment'].astype(str).str.lower().str.strip()
            df['sentiment'] = df['sentiment'].replace({
                'pos': 'positive', 'p': 'positive', '1': 'positive', 'true': 'positive', 'yes': 'positive',
                'neg': 'negative', 'n': 'negative', '0': 'negative', 'false': 'negative', 'no': 'negative'
            })
            df['sentiment_encoded'] = df['sentiment'].map({'positive': 1, 'negative': 0})
            
        df = df[df['sentiment_encoded'].notna()].reset_index(drop=True)
        
        final_rows = len(df)
        mlflow.log_metric("initial_rows", initial_rows)
        mlflow.log_metric("final_rows_preprocessed", final_rows)

        # --- 5. Splitting dan Saving ---
        X = df['review']
        y = df['sentiment_encoded'].astype(int)
        
        if y.nunique() < 2:
            raise ValueError("Target memiliki kurang dari 2 kelas. Tidak dapat melakukan stratified split.")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        df_train = pd.DataFrame({'review': X_train.values, 'sentiment': y_train.values})
        df_train['split'] = 'train'
        df_test = pd.DataFrame({'review': X_test.values, 'sentiment': y_test.values})
        df_test['split'] = 'test'
        df_preprocessed = pd.concat([df_train, df_test], ignore_index=True)

        output_folder_path = Path(output_dir)
        output_folder_path.mkdir(parents=True, exist_ok=True)
        
        output_file = output_folder_path / 'amazon_preprocessed.csv'
        
        df_preprocessed.to_csv(output_file, index=False, encoding='utf-8') 
        
        # LOGGING DATA PREPROCESSED SEBAGAI ARTEFAK
        mlflow.log_artifact(str(output_file), artifact_path="preprocessed_data")

        print(f"Preprocessing selesai. Data siap dilatih disimpan di: {output_file.resolve()}")
        print(f"MLflow Preprocessing Run ID: {mlflow.active_run().info.run_id}")
    
    return df_preprocessed

# Bagian eksekusi script
if __name__ == '__main__':
    data_mentah_path = Path(r"C:\Users\FathurNitro\OneDrive\Documents\sub dicoding\membangun machine learning bismillah\amazon.csv") 
    
    try:
        data_siap_latih = automate_preprocessing_pipeline(
            input_file_path=data_mentah_path,
            output_dir='amazon_preprocessing'
        )
        if data_siap_latih is not None:
            print("\nContoh 5 baris data siap latih:")
            # Hapus .head() untuk menghindari cetakan yang bermasalah di konsol, 
            # atau biarkan jika Anda ingin melihat output yang kemungkinan berisi emoji
            print(data_siap_latih[['review', 'sentiment', 'split']].head()) 
    except Exception as e:
        # Perbaiki handling error, gunakan sys.exit(1) hanya jika terjadi error fatal
        if "Gagal total memuat" in str(e) or "File not found" in str(e) or "No valid text column" in str(e) or "Target memiliki kurang dari 2 kelas" in str(e):
             print(f"\nFATAL ERROR during automation: {e}")
             sys.exit(1)
        else:
             # Ini adalah error cetak konsol yang tidak fatal untuk proses ML
             print(f"\nWARNING: Error pencetakan konsol non-fatal dicegah: {e}")
             # Lanjutkan eksekusi tanpa keluar
