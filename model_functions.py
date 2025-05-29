import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Add, BatchNormalization, Activation
import matplotlib
matplotlib.use('Agg') # Menggunakan backend Agg untuk lingkungan non-GUI seperti Flask
import matplotlib.pyplot as plt
import os

# --- Fungsi Pra-pemrosesan Data ---
def load_and_preprocess_data(file_path, test_size_ratio, random_state_val):
    """
    Memuat data, melakukan pra-pemrosesan, seleksi fitur, dan pemisahan data.
    Mengembalikan X_train, X_test, y_train, y_test, scaler, label_encoders, dan selected_feature_names.
    """
    df = pd.read_csv(file_path)
    df_clean = df.copy()

    # Menghapus kolom yang tidak diperlukan jika ada (contoh dari notebook, EmployeeCount, EmployeeNumber, StandardHours, Over18)
    # Kolom-kolom ini seringkali unik atau tidak memiliki varians
    cols_to_drop = ['EmployeeCount', 'EmployeeNumber', 'StandardHours', 'Over18']
    # Cek apakah kolom ada sebelum di-drop
    existing_cols_to_drop = [col for col in cols_to_drop if col in df_clean.columns]
    if existing_cols_to_drop:
        df_clean = df_clean.drop(columns=existing_cols_to_drop)


    # Penanganan Outlier (menggunakan IQR dan mengganti dengan median)
    numeric_cols = df_clean.select_dtypes(include=np.number).columns.tolist()
    
    # Hapus target dari numeric_cols jika ada (meskipun Attrition akan di-encode nanti)
    if 'Attrition' in numeric_cols: # Attrition akan di-encode, jadi tidak dianggap numerik murni di sini
        numeric_cols.remove('Attrition')


    for col in numeric_cols:
        if col in df_clean.columns: # Pastikan kolom masih ada
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Ganti outliers dengan NaN terlebih dahulu
            df_clean[col] = np.where((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound), np.nan, df_clean[col])
            # Isi NaN (termasuk yang dari outliers) dengan median
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # Encoding Fitur Kategorikal
    label_encoders = {}
    data_fe = df_clean.copy()
    object_cols = [col for col in data_fe.columns if data_fe[col].dtype == 'object']

    for col in object_cols:
        le = LabelEncoder()
        data_fe[col] = le.fit_transform(data_fe[col])
        label_encoders[col] = le # Simpan encoder

    # Pisahkan fitur (X) dan target (y)
    # Pastikan 'Attrition' adalah nama kolom target yang sudah di-encode
    if 'Attrition' not in data_fe.columns:
        raise ValueError("Kolom target 'Attrition' tidak ditemukan setelah encoding.")
        
    X = data_fe.drop(columns='Attrition')
    y = data_fe['Attrition']

    # Seleksi Fitur menggunakan Chi2
    # Chi2 memerlukan nilai non-negatif. Data sudah di-encode dan outlier ditangani.
    if (X < 0).any().any():
        print("Peringatan: Ada nilai negatif di X sebelum Chi2. Menggunakan abs().")
        X_non_negative = X.abs() 
    else:
        X_non_negative = X

    chi_square_values = chi2(X_non_negative, y)
    p_values_series = pd.Series(chi_square_values[1], index=X.columns)
    
    # Menggunakan p_value < 0.05 sebagai standar, bukan < 2 seperti di notebook
    # Jika ingin mengikuti notebook, ubah 0.05 menjadi 2.0
    selected_features_mask = p_values_series < 0.05 
    selected_feature_names = X.columns[selected_features_mask].tolist()

    if not selected_feature_names:
        print("Peringatan: Tidak ada fitur yang terpilih dengan p-value < 0.05. Menggunakan semua fitur.")
        X_fs = X.copy()
        selected_feature_names = X.columns.tolist()
    else:
        X_fs = X[selected_feature_names]

    # Normalisasi dengan MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_fs_sc_array = scaler.fit_transform(X_fs)
    X_fs_sc = pd.DataFrame(X_fs_sc_array, columns=selected_feature_names, index=X_fs.index)

    # Pembagian data
    X_train, X_test, y_train, y_test = train_test_split(X_fs_sc, y, test_size=test_size_ratio, random_state=random_state_val, stratify=y)
    
    return X_train, X_test, y_train, y_test, scaler, label_encoders, selected_feature_names

# --- Definisi Arsitektur Model ---
def model_densenet_custom(input_dim):
    """Arsitektur mirip DenseNet (Custom) dari notebook."""
    inputs = Input(shape=(input_dim,))
    x1 = Dense(64, activation='relu')(inputs)
    x2 = Dense(64, activation='relu')(x1)
    concat1 = Concatenate()([x1, x2])
    x3 = Dense(64, activation='relu')(concat1)
    concat2 = Concatenate()([concat1, x3])
    x4 = Dense(32, activation='relu')(concat2)
    outputs = Dense(1, activation='sigmoid')(x4) # Binary classification
    model = Model(inputs=inputs, outputs=outputs)
    return model

def model_resmlp_custom(input_dim):
    """Arsitektur mirip ResMLP (Custom) dari notebook."""
    inputs = Input(shape=(input_dim,))
    # Block 1
    x = Dense(64)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    res1 = x 
    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Add()([x, res1])
    # Block 2
    res2 = x
    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Add()([x, res2])
    outputs = Dense(1, activation='sigmoid')(x) # Binary classification
    model = Model(inputs=inputs, outputs=outputs)
    return model

def model_resnet_custom(input_dim):
    """Arsitektur mirip ResNet (Custom) dari notebook."""
    def residual_block(x_input, units):
        shortcut = x_input
        # Cek apakah perlu Dense layer untuk menyesuaikan dimensi shortcut
        if x_input.shape[-1] != units:
            shortcut = Dense(units, activation='linear')(shortcut) # Linear activation untuk penyesuaian dimensi

        x_res = Dense(units)(x_input)
        x_res = BatchNormalization()(x_res)
        x_res = Activation('relu')(x_res)
        x_res = Dense(units)(x_res)
        x_res = BatchNormalization()(x_res)
        
        x_res = Add()([shortcut, x_res])
        x_res = Activation('relu')(x_res)
        return x_res

    inputs = Input(shape=(input_dim,))
    x = Dense(64, activation='relu')(inputs) # Lapisan awal untuk memproyeksikan input_dim ke 64
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = residual_block(x, 64)
    x = residual_block(x, 64)

    x = Dense(32, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x) # Binary classification
    model = Model(inputs=inputs, outputs=outputs)
    return model

# --- Fungsi untuk Plotting ---
def plot_training_history(history, optimizer_name, architecture_name, split_ratio_name, static_folder='static/images'):
    """Menyimpan plot training & validation accuracy/loss."""
    if not os.path.exists(static_folder):
        os.makedirs(static_folder)

    plt.figure(figsize=(12, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Accuracy ({architecture_name}, {optimizer_name}, {split_ratio_name})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Loss ({architecture_name}, {optimizer_name}, {split_ratio_name})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plot_filename = f"plot_{architecture_name}_{optimizer_name}_{split_ratio_name.replace(':', '_')}.png"
    plot_path = os.path.join(static_folder, plot_filename)
    
    try:
        plt.savefig(plot_path)
        print(f"Plot disimpan di: {plot_path}")
    except Exception as e:
        print(f"Error saat menyimpan plot: {e}")
    plt.close() # Tutup plot agar tidak mengkonsumsi memori
    return plot_filename

# --- Fungsi untuk memproses input pengguna untuk prediksi ---
def preprocess_input_for_prediction(input_data_str, original_df_columns, selected_feature_names, label_encoders, scaler):
    """
    Memproses string input dari pengguna menjadi format yang siap untuk prediksi.
    original_df_columns: daftar nama kolom dari DataFrame asli sebelum encoding (kecuali target dan kolom yang di-drop).
    """
    # Hapus kolom target dan kolom yang di-drop dari original_df_columns jika ada
    # Kolom yang di-drop di load_and_preprocess_data: 'EmployeeCount', 'EmployeeNumber', 'StandardHours', 'Over18'
    cols_to_exclude_from_input = ['Attrition', 'EmployeeCount', 'EmployeeNumber', 'StandardHours', 'Over18']
    input_feature_order = [col for col in original_df_columns if col not in cols_to_exclude_from_input]


    if not input_feature_order:
         raise ValueError("Tidak dapat menentukan urutan fitur input. original_df_columns mungkin kosong atau salah.")

    try:
        # Asumsi input_data_str adalah comma-separated values
        values = [val.strip() for val in input_data_str.split(',')]
        if len(values) != len(input_feature_order):
            raise ValueError(f"Jumlah input ({len(values)}) tidak sesuai dengan jumlah fitur yang diharapkan ({len(input_feature_order)}). Fitur yang diharapkan: {', '.join(input_feature_order)}")

        input_df = pd.DataFrame([values], columns=input_feature_order)

        # Konversi tipe data dan Label Encoding
        for col in input_df.columns:
            if col in label_encoders: # Jika kolom adalah kategorikal yang di-encode
                try:
                    # Coba transform, jika error (kategori baru), tangani atau beri tahu user
                    input_df[col] = label_encoders[col].transform(input_df[col])
                except ValueError as e: # Kategori tidak dikenal
                    # Coba handle dengan nilai default (misal, nilai modus dari training set atau -1)
                    # Untuk kesederhanaan, kita akan raise error atau bisa juga memberi tahu user
                    # bahwa kategori tidak valid.
                    # Alternatif: le.classes_ akan memberi tahu kategori yang valid.
                    valid_categories = list(label_encoders[col].classes_)
                    raise ValueError(f"Nilai '{input_df[col].iloc[0]}' untuk fitur '{col}' tidak valid. Kategori yang valid: {valid_categories}. Error: {e}")
            else: # Jika kolom numerik, coba konversi ke float
                try:
                    input_df[col] = pd.to_numeric(input_df[col])
                except ValueError:
                    raise ValueError(f"Nilai '{input_df[col].iloc[0]}' untuk fitur numerik '{col}' tidak dapat dikonversi ke angka.")
        
        # Pastikan semua kolom numerik yang diharapkan ada dan bertipe numerik
        # (setelah encoding, semua fitur yang masuk ke scaler seharusnya numerik)
        for col in selected_feature_names:
            if col not in input_df.columns:
                 raise ValueError(f"Fitur yang diseleksi '{col}' tidak ditemukan dalam data input setelah pra-pemrosesan awal.")
            if not pd.api.types.is_numeric_dtype(input_df[col]):
                raise ValueError(f"Fitur '{col}' seharusnya numerik setelah encoding, tetapi tipenya {input_df[col].dtype}.")


        # Pilih fitur yang sesuai dengan model (selected_feature_names)
        input_fs = input_df[selected_feature_names]

        # Scaling
        input_scaled_array = scaler.transform(input_fs)
        # input_scaled_df = pd.DataFrame(input_scaled_array, columns=selected_feature_names) # Tidak perlu DataFrame di sini

        return input_scaled_array # Return sebagai numpy array

    except Exception as e:
        # Tambahkan logging atau print untuk debugging
        print(f"Error di preprocess_input_for_prediction: {e}")
        # Re-raise error agar bisa ditangkap di app.py
        raise

if __name__ == '__main__':
    # Contoh penggunaan (untuk testing fungsi secara mandiri)
    file_path = 'WA_Fn-UseC_-HR-Employee-Attrition.csv'
    if not os.path.exists(file_path):
        print(f"File dataset {file_path} tidak ditemukan. Pastikan file ada di direktori yang sama.")
    else:
        print("Testing load_and_preprocess_data...")
        try:
            X_train, X_test, y_train, y_test, scaler, label_encoders, selected_features = \
                load_and_preprocess_data(file_path, test_size_ratio=0.2, random_state_val=42)
            
            print("X_train shape:", X_train.shape)
            print("X_test shape:", X_test.shape)
            print("y_train shape:", y_train.shape)
            print("y_test shape:", y_test.shape)
            print("Jumlah fitur terpilih:", len(selected_features))
            print("Fitur terpilih:", selected_features)
            print("Label encoders ditemukan untuk kolom:", list(label_encoders.keys()))

            # Test model definition
            input_dim = X_train.shape[1]
            model1 = model_densenet_custom(input_dim)
            model1.summary()

            # Dapatkan kolom asli untuk input prediksi
            original_df_for_columns = pd.read_csv(file_path)
            original_cols = original_df_for_columns.columns.tolist()

            # Contoh input string (sesuaikan jumlah dan urutan dengan fitur asli sebelum encoding & drop)
            # Ini adalah contoh, Anda perlu menyesuaikan dengan fitur aktual dataset Anda
            # Misal, jika dataset asli punya 30 fitur setelah drop kolom tidak penting, maka ada 30 nilai di sini.
            # Urutan harus sama dengan df.columns sebelum 'Attrition' dan sebelum encoding.
            # Contoh: df_clean.drop(columns=['Attrition', 'EmployeeCount', ...]).columns
            
            # Dapatkan urutan fitur input yang diharapkan
            cols_to_exclude_from_input = ['Attrition', 'EmployeeCount', 'EmployeeNumber', 'StandardHours', 'Over18']
            expected_input_features_ordered = [col for col in original_cols if col not in cols_to_exclude_from_input]
            print(f"\nUrutan fitur yang diharapkan untuk input prediksi ({len(expected_input_features_ordered)} fitur): {', '.join(expected_input_features_ordered)}")
            
            # Buat contoh data input string berdasarkan fitur pertama dari dataset asli
            if not X_train.empty: # Pastikan X_train tidak kosong
                # Ambil data baris pertama dari CSV asli untuk contoh input
                sample_raw_data_series = original_df_for_columns.iloc[0]
                
                # Buat string input dari sample_raw_data_series sesuai urutan expected_input_features_ordered
                example_input_values = []
                for col_name in expected_input_features_ordered:
                    example_input_values.append(str(sample_raw_data_series[col_name]))
                example_input_str = ", ".join(example_input_values)

                print(f"\nContoh string input untuk prediksi (berdasarkan baris pertama data asli):\n{example_input_str}")

                print("\nTesting preprocess_input_for_prediction...")
                try:
                    processed_input = preprocess_input_for_prediction(
                        example_input_str,
                        original_cols, # Kolom dari df asli
                        selected_features,
                        label_encoders,
                        scaler
                    )
                    print("Input yang diproses untuk prediksi shape:", processed_input.shape)
                    print("Input yang diproses (sampel):", processed_input[0][:5]) # Tampilkan 5 elemen pertama
                except Exception as e_pred_proc:
                    print(f"Error saat testing preprocess_input_for_prediction: {e_pred_proc}")
            else:
                print("Tidak dapat membuat contoh input karena X_train kosong setelah pra-pemrosesan.")


        except Exception as e_main_test:
            print(f"Error saat testing utama: {e_main_test}")

