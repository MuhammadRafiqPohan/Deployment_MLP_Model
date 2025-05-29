Cara menjalan Deployment :

unduh semua code dan csv yang ada
buat folder khusus untuk deployment (misalnya, "Tugas_Deployment")
pada terminal vscode ketik "python -m venv .venv"
kemudian aktikan dengan ketik ".venv\Scripts\activate"
lakukan instalasi Pustaka dengan ketik "pip install Flask pandas scikit-learn tensorflow matplotlib seaborn numpy" pada terminal
Di terminal, dengan virtual environment aktif, jalankan "pip install -r requirements.txt" Ini akan menginstal semua pustaka yang tercantum di requirements.txt
Jalankan Aplikasi Flask, di terminal pastikan sedang berada di dalam folder Tugas_Deployment (di mana app.py berada), lalu jalankan: python app.py
Buka browser web dan ketik "http://127.0.0.1:5000/"
Lakukan Uji Coba
