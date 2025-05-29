from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import os
import numpy as np

# Impor fungsi dari model_functions.py
from model_functions import (
    load_and_preprocess_data,
    model_densenet_custom,
    model_resmlp_custom,
    model_resnet_custom,
    plot_training_history,
    preprocess_input_for_prediction
)

app = Flask(__name__)
app.secret_key = "supersecretkey" 

# Path ke dataset
DATASET_PATH = 'WA_Fn-UseC_-HR-Employee-Attrition.csv'

# Pastikan folder static/images ada
STATIC_FOLDER = os.path.join('static')
IMAGES_FOLDER = os.path.join(STATIC_FOLDER, 'images')
if not os.path.exists(IMAGES_FOLDER):
    os.makedirs(IMAGES_FOLDER)

# Global variables for feature details
FEATURE_DETAILS_LIST = []
ORIGINAL_DF_COLUMNS = [] 

# --- Penjelasan Fitur dalam Bahasa Indonesia ---
ABOUT_DATASET_TEXT_ID = "Dataset ini bertujuan untuk mengungkap faktor-faktor yang menyebabkan atrisi karyawan (karyawan keluar dari perusahaan). Anda dapat mengeksplorasi pertanyaan-pertanyaan penting seperti ‘tunjukkan rincian jarak dari rumah berdasarkan peran pekerjaan dan status atrisi’ atau ‘bandingkan rata-rata pendapatan bulanan berdasarkan tingkat pendidikan dan status atrisi’. Ini adalah dataset fiktif yang dibuat oleh para ilmuwan data IBM."

FEATURE_LABELS_ID = {
    "Age": "Usia",
    "BusinessTravel": "Perjalanan Bisnis",
    "DailyRate": "Tarif Harian",
    "Department": "Departemen",
    "DistanceFromHome": "Jarak dari Rumah",
    "Education": "Pendidikan",
    "EducationField": "Bidang Pendidikan",
    "EnvironmentSatisfaction": "Kepuasan Lingkungan",
    "Gender": "Jenis Kelamin",
    "HourlyRate": "Tarif Per Jam",
    "JobInvolvement": "Keterlibatan Kerja",
    "JobLevel": "Tingkat Jabatan",
    "JobRole": "Peran Pekerjaan",
    "JobSatisfaction": "Kepuasan Kerja",
    "MaritalStatus": "Status Perkawinan",
    "MonthlyIncome": "Pendapatan Bulanan",
    "MonthlyRate": "Tarif Bulanan",
    "NumCompaniesWorked": "Jumlah Perusahaan Sebelumnya",
    "OverTime": "Lembur",
    "PercentSalaryHike": "Persentase Kenaikan Gaji",
    "PerformanceRating": "Peringkat Kinerja",
    "RelationshipSatisfaction": "Kepuasan Hubungan",
    "StockOptionLevel": "Tingkat Opsi Saham",
    "TotalWorkingYears": "Total Tahun Bekerja",
    "TrainingTimesLastYear": "Frekuensi Pelatihan Tahun Lalu",
    "WorkLifeBalance": "Keseimbangan Kehidupan Kerja",
    "YearsAtCompany": "Tahun di Perusahaan",
    "YearsInCurrentRole": "Tahun di Peran Saat Ini",
    "YearsSinceLastPromotion": "Tahun Sejak Promosi Terakhir",
    "YearsWithCurrManager": "Tahun dengan Manajer Saat Ini"
}

FEATURE_EXPLANATIONS_ID = {
    "Age": "Usia karyawan dalam tahun.",
    "Attrition": "Apakah karyawan keluar dari perusahaan (Ya) atau tidak (Tidak). (Ini adalah target prediksi)",
    "BusinessTravel": "Seberapa sering karyawan melakukan perjalanan bisnis (Tidak Pernah, Jarang, Sering).",
    "DailyRate": "Tarif harian gaji karyawan (angka).",
    "Department": "Departemen tempat karyawan bekerja (misalnya, Penjualan, Penelitian & Pengembangan, Sumber Daya Manusia).",
    "DistanceFromHome": "Jarak dari rumah ke tempat kerja dalam kilometer.",
    "Education": {
        "description": "Tingkat pendidikan karyawan.",
        "map": {1: 'Di Bawah Perguruan Tinggi', 2: 'Perguruan Tinggi/Akademi', 3: 'Sarjana (S1)', 4: 'Magister (S2)', 5: 'Doktor (S3)'}
    },
    "EducationField": "Bidang studi pendidikan karyawan (misalnya, Ilmu Hayati, Medis, Pemasaran).",
    "EmployeeCount": "Jumlah karyawan (biasanya selalu 1 untuk setiap baris data, seringkali tidak relevan untuk model).",
    "EmployeeNumber": "Nomor identifikasi unik karyawan (biasanya tidak relevan untuk model).",
    "EnvironmentSatisfaction": {
        "description": "Tingkat kepuasan karyawan terhadap lingkungan kerja.",
        "map": {1: 'Rendah', 2: 'Sedang', 3: 'Tinggi', 4: 'Sangat Tinggi'}
    },
    "Gender": "Jenis kelamin karyawan (Pria, Wanita).",
    "HourlyRate": "Tarif per jam gaji karyawan (angka).",
    "JobInvolvement": {
        "description": "Tingkat keterlibatan karyawan dalam pekerjaan.",
        "map": {1: 'Rendah', 2: 'Sedang', 3: 'Tinggi', 4: 'Sangat Tinggi'}
    },
    "JobLevel": "Tingkat jabatan karyawan dalam perusahaan (biasanya 1 hingga 5, semakin tinggi semakin senior).",
    "JobRole": "Peran atau jabatan spesifik karyawan (misalnya, Eksekutif Penjualan, Ilmuwan Riset).",
    "JobSatisfaction": {
        "description": "Tingkat kepuasan karyawan terhadap pekerjaan.",
        "map": {1: 'Rendah', 2: 'Sedang', 3: 'Tinggi', 4: 'Sangat Tinggi'}
    },
    "MaritalStatus": "Status perkawinan karyawan (Lajang, Menikah, Bercerai).",
    "MonthlyIncome": "Pendapatan bulanan karyawan.",
    "MonthlyRate": "Tarif bulanan (angka, mungkin terkait dengan kompensasi atau biaya lainnya).",
    "NumCompaniesWorked": "Jumlah perusahaan tempat karyawan pernah bekerja sebelumnya.",
    "Over18": "Apakah karyawan berusia di atas 18 tahun (biasanya selalu 'Y' atau Ya, seringkali tidak relevan).",
    "OverTime": "Apakah karyawan bekerja lembur (Ya) atau tidak (Tidak).",
    "PercentSalaryHike": "Persentase kenaikan gaji karyawan pada tahun terakhir.",
    "PerformanceRating": {
        "description": "Penilaian kinerja karyawan oleh perusahaan.",
        "map": {1: 'Rendah', 2: 'Baik', 3: 'Sangat Baik', 4: 'Luar Biasa'} 
    },
    "RelationshipSatisfaction": {
        "description": "Tingkat kepuasan karyawan terhadap hubungan interpersonal di tempat kerja.",
        "map": {1: 'Rendah', 2: 'Sedang', 3: 'Tinggi', 4: 'Sangat Tinggi'}
    },
    "StandardHours": "Jam kerja standar karyawan (biasanya selalu 80, seringkali tidak relevan).",
    "StockOptionLevel": "Tingkat opsi saham yang dimiliki karyawan (0 hingga 3).",
    "TotalWorkingYears": "Total tahun pengalaman kerja karyawan secara keseluruhan.",
    "TrainingTimesLastYear": "Berapa kali karyawan mengikuti pelatihan pada tahun lalu.",
    "WorkLifeBalance": {
        "description": "Keseimbangan antara kehidupan kerja dan pribadi karyawan.",
        "map": {1: 'Buruk', 2: 'Baik', 3: 'Lebih Baik', 4: 'Terbaik'}
    },
    "YearsAtCompany": "Jumlah tahun karyawan bekerja di perusahaan saat ini.",
    "YearsInCurrentRole": "Jumlah tahun karyawan berada di peran/jabatan saat ini.",
    "YearsSinceLastPromotion": "Jumlah tahun sejak promosi terakhir karyawan.",
    "YearsWithCurrManager": "Jumlah tahun karyawan bekerja dengan manajer saat ini."
}

FALLBACK_EXPECTED_FEATURES = ['Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome', 'Education', 'EducationField', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']
FALLBACK_EXAMPLE_VALUES = ['41', 'Travel_Rarely', '1102', 'Sales', '1', '2', 'Life Sciences', '2', 'Female', '94', '3', '2', 'Sales Executive', '4', 'Single', '5993', '19479', '8', 'Yes', '11', '3', '1', '0', '8', '0', '1', '6', '4', '0', '5']


def initialize_feature_details():
    global FEATURE_DETAILS_LIST, ORIGINAL_DF_COLUMNS 
    temp_feature_details = []
    original_df_columns_local = []
    
    try:
        if os.path.exists(DATASET_PATH):
            original_df = pd.read_csv(DATASET_PATH)
            original_df_columns_local = original_df.columns.tolist()
            ORIGINAL_DF_COLUMNS = original_df_columns_local 

            cols_to_exclude = ['Attrition', 'EmployeeCount', 'EmployeeNumber', 'StandardHours', 'Over18']
            expected_input_features_ordered_for_form = [col for col in original_df_columns_local if col not in cols_to_exclude]

            if not original_df.empty:
                first_row_examples = {col: str(original_df.iloc[0][col]) for col in expected_input_features_ordered_for_form}

                for col_name in expected_input_features_ordered_for_form: 
                    detail = {
                        'name': col_name, 
                        'label_id': FEATURE_LABELS_ID.get(col_name, col_name), 
                        'example': first_row_examples.get(col_name, '')
                    }
                    col_dtype = original_df[col_name].dtype
                    unique_values = original_df[col_name].unique()

                    explanation_data = FEATURE_EXPLANATIONS_ID.get(col_name) 
                    if isinstance(explanation_data, dict) and 'map' in explanation_data:
                        detail['type'] = 'select'
                        detail['options'] = [str(k) for k in explanation_data['map'].keys()]
                        if not detail['example'] and detail['options']:
                             detail['example'] = detail['options'][0]
                    elif col_dtype == 'object':
                        if len(unique_values) < 20 : 
                            detail['type'] = 'select'
                            detail['options'] = sorted([str(uv) for uv in unique_values if pd.notna(uv)])
                        else: 
                            detail['type'] = 'text' 
                    elif pd.api.types.is_numeric_dtype(col_dtype):
                        ordinal_like_cols_max_val = { 
                            'JobLevel': 5, 'StockOptionLevel': 3 
                        }
                        if col_name in ordinal_like_cols_max_val and not (isinstance(explanation_data, dict) and 'map' in explanation_data):
                            detail['type'] = 'select' 
                            detail['options'] = [str(i) for i in range(1, ordinal_like_cols_max_val[col_name] + 1)]
                        else: 
                            detail['type'] = 'number'
                            min_val = original_df[col_name].min()
                            max_val = original_df[col_name].max()
                            if pd.notna(min_val): detail['min'] = min_val
                            if pd.notna(max_val): detail['max'] = max_val
                    else: 
                        detail['type'] = 'text'
                    temp_feature_details.append(detail)
            else:
                print("Peringatan: DataFrame asli kosong, contoh nilai tidak dapat dibuat.")
                for i, name in enumerate(FALLBACK_EXPECTED_FEATURES): 
                    temp_feature_details.append({
                        'name': name, 
                        'label_id': FEATURE_LABELS_ID.get(name, name),
                        'type': 'text', 
                        'example': FALLBACK_EXAMPLE_VALUES[i]
                    })
        else:
            print(f"PERINGATAN: File dataset '{DATASET_PATH}' tidak ditemukan. Menggunakan fallback untuk fitur input.")
            ORIGINAL_DF_COLUMNS = [] 
            for i, name in enumerate(FALLBACK_EXPECTED_FEATURES): 
                temp_feature_details.append({
                    'name': name, 
                    'label_id': FEATURE_LABELS_ID.get(name, name),
                    'type': 'text', 
                    'example': FALLBACK_EXAMPLE_VALUES[i]
                })
        
        FEATURE_DETAILS_LIST = temp_feature_details
    except Exception as e:
        print(f"Error saat inisialisasi detail fitur: {e}")
        FEATURE_DETAILS_LIST.clear() 
        ORIGINAL_DF_COLUMNS = [] 
        for i, name in enumerate(FALLBACK_EXPECTED_FEATURES): 
            FEATURE_DETAILS_LIST.append({
                'name': name, 
                'label_id': FEATURE_LABELS_ID.get(name, name),
                'type': 'text', 
                'example': FALLBACK_EXAMPLE_VALUES[i]
            })

initialize_feature_details() 

@app.route('/', methods=['GET', 'POST'])
def index():
    global ORIGINAL_DF_COLUMNS 
    current_expected_features = [f['name'] for f in FEATURE_DETAILS_LIST]
    
    if request.method == 'POST':
        try:
            split_ratio_str = request.form['split_ratio']
            architecture_choice = request.form['architecture']
            optimizer_choice = request.form['optimizer']

            user_input_values = []
            all_inputs_empty = True
            for feature_detail in FEATURE_DETAILS_LIST: 
                feature_name = feature_detail['name']
                value = request.form.get(f'feature_{feature_name}', '').strip()
                user_input_values.append(value)
                if value:
                    all_inputs_empty = False
            
            user_input_data_str = ""
            if not all_inputs_empty:
                user_input_data_str = ",".join(user_input_values)
            else:
                print("Semua field input karyawan kosong, tidak ada prediksi yang akan dilakukan.")

            if split_ratio_str == "80:20": test_size, split_ratio_name_file = 0.2, "80_20"
            elif split_ratio_str == "70:30": test_size, split_ratio_name_file = 0.3, "70_30"
            elif split_ratio_str == "90:10": test_size, split_ratio_name_file = 0.1, "90_10"
            else: return "Rasio pembagian data tidak valid", 400

            X_train, X_test, y_train, y_test, scaler, label_encoders, selected_feature_names = \
                load_and_preprocess_data(DATASET_PATH, test_size_ratio=test_size, random_state_val=42)

            if X_train.empty or X_test.empty:
                return "Error: Data training atau testing kosong setelah pra-pemrosesan.", 500
            
            input_dim = X_train.shape[1]

            if architecture_choice == 'DenseNetCustom': model = model_densenet_custom(input_dim)
            elif architecture_choice == 'ResMLPCustom': model = model_resmlp_custom(input_dim)
            elif architecture_choice == 'ResNetCustom': model = model_resnet_custom(input_dim)
            else: return "Arsitektur tidak valid", 400

            learning_rate = 0.001 
            if optimizer_choice == 'adam': opt = Adam(learning_rate=learning_rate)
            elif optimizer_choice == 'sgd': opt = SGD(learning_rate=learning_rate)
            elif optimizer_choice == 'RMSprop': opt = RMSprop(learning_rate=learning_rate)
            else: return "Optimizer tidak valid", 400

            model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
            history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32, verbose=0)
            loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
            accuracy_percent = accuracy * 100
            plot_filename = plot_training_history(history, optimizer_choice, architecture_choice, split_ratio_name_file, static_folder=IMAGES_FOLDER)

            prediction_text = "Input karyawan tidak disediakan atau semua field kosong."
            if user_input_data_str:
                try:
                    if not ORIGINAL_DF_COLUMNS:
                        print("PERINGATAN: ORIGINAL_DF_COLUMNS kosong sebelum prediksi, mencoba inisialisasi ulang...")
                        initialize_feature_details() 
                        if not ORIGINAL_DF_COLUMNS: 
                             raise FileNotFoundError("Tidak dapat menentukan kolom asli untuk pra-pemrosesan input setelah upaya inisialisasi ulang.")

                    processed_input_for_pred = preprocess_input_for_prediction(
                        user_input_data_str, ORIGINAL_DF_COLUMNS, 
                        selected_feature_names, label_encoders, scaler
                    )
                    prediction_prob = model.predict(processed_input_for_pred)[0][0]
                    # Mengubah format output prediksi
                    attrition_status = "Ya" if prediction_prob > 0.5 else "Tidak"
                    prediction_text = f"Attrition: {attrition_status} (Probabilitas Atrisi: {prediction_prob:.4f})"
                except ValueError as ve: 
                     prediction_text = f"Error saat memproses input: {str(ve)}."
                except Exception as e_pred:
                    prediction_text = f"Error saat prediksi: {str(e_pred)}"
            
            return render_template('result.html',
                                   accuracy=f"{accuracy_percent:.2f}%",
                                   plot_image_filename=plot_filename, 
                                   prediction_result=prediction_text,
                                   selected_split_ratio=split_ratio_str,
                                   selected_architecture=architecture_choice,
                                   selected_optimizer=optimizer_choice,
                                   user_submitted_input_str=user_input_data_str, 
                                   input_features_labels_list=[f['label_id'] for f in FEATURE_DETAILS_LIST] 
                                   )
        except FileNotFoundError:
            return f"Error: File dataset '{DATASET_PATH}' tidak ditemukan atau tidak dapat diakses.", 500
        except ValueError as ve: 
             return f"Terjadi kesalahan input atau validasi: {str(ve)}", 400
        except Exception as e:
            import traceback
            print(f"Terjadi error tidak terduga: {e}")
            print(traceback.format_exc())
            return f"Terjadi error tidak terduga pada server: {str(e)}", 500

    return render_template('index.html', 
                           feature_details_list=FEATURE_DETAILS_LIST, 
                           feature_explanations=FEATURE_EXPLANATIONS_ID, 
                           about_dataset_text=ABOUT_DATASET_TEXT_ID) 

if __name__ == '__main__':
    if not os.path.exists(DATASET_PATH) and not ORIGINAL_DF_COLUMNS: 
        print(f"FATAL: File dataset '{DATASET_PATH}' tidak ditemukan dan fallback fitur gagal. Aplikasi mungkin tidak berfungsi dengan benar.")
    elif not os.path.exists(DATASET_PATH):
        print(f"PERINGATAN: File dataset '{DATASET_PATH}' tidak ditemukan. Aplikasi berjalan dengan fitur input fallback.")
    app.run(debug=True)
