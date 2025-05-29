Cara menjalan Deployment :

1. unduh semua code dan csv yang ada
2. buat folder khusus untuk deployment (misalnya, "Tugas_Deployment")
3. pada terminal vscode ketik "python -m venv .venv"
4. kemudian aktikan dengan ketik ".venv\Scripts\activate"
5. lakukan instalasi Pustaka dengan ketik "pip install Flask pandas scikit-learn tensorflow matplotlib seaborn numpy" pada terminal
6. Di terminal, dengan virtual environment aktif, jalankan "pip install -r requirements.txt" Ini akan menginstal semua pustaka yang tercantum di requirements.txt
7. Jalankan Aplikasi Flask, di terminal pastikan sedang berada di dalam folder Tugas_Deployment (di mana app.py berada), lalu jalankan: python app.py
8. Buka browser web dan ketik "http://127.0.0.1:5000/"
9. Lakukan Uji Coba
