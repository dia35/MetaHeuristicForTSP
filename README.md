╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌   1 # TSP Genetik Algoritma — V6 Time & Iteration Unified Runner
   2
   3 **Öğrenci:** Asrın Çoban — 23255603
   4 **Ders:** Sezgisel Yöntemler
   5
   6 ---
   7
   8 ## Proje Hakkında
   9
  10 Bu proje, **Gezgin Satıcı Problemi (TSP)**'ni Genetik Algoritma (GA) ile çözmek için geliştirilmiş çok aşamalı b
     ir deney çerçevesidir. 9 farklı TSP veri seti üzerinde parametre, seçim, çaprazlama ve mutasyon operatörlerini s
     istematik olarak karşılaştırır.
  11
  12 ---
  13
  14 ## Dosyalar
  15
  16 | Dosya | Açıklama |
  17 |-------|----------|
  18 | `v6_time_iteration_unified_runner.py` | Ana Python scripti — tüm pipeline bu dosyada |
  19 | `v6_time_iteration_unified_results.xlsx` | Tüm aşama sonuçlarını içeren Excel çıktısı |
  20 | `TSP_GA_V6_TimeIteration_Unified_Final_Rapor.pdf` | Nihai proje raporu |
  21
  22 ---
  23
  24 ## Kullanılan Veri Setleri
  25
  26 `berlin52`, `ch130`, `d493`, `eil101`, `eil51`, `eil76`, `kroA100`, `pcb442`, `pr299`
  27 Seed'ler: `11, 42, 123, 2026, 9999`
  28
  29 ---
  30
  31 ## Pipeline Aşamaları
  32
  33 | Aşama | Açıklama | Run Sayısı | Süre Limiti | İterasyon Limiti |
  34 |-------|----------|-----------|-------------|-----------------|
  35 | 1 — Parametre Grid | 48 farklı parametre kombinasyonu | 2160 | 1 sn | 5.000 |
  36 | 2 — Top10 Onay | En iyi 10 parametrenin doğrulanması | 450 | 5 sn | 20.000 |
  37 | 3 — Seçim | Roulette vs Tournament (ts=2,3,5,7,10) | 270 | 5 sn | 20.000 |
  38 | 4 — Çaprazlama | PMX, OX, SCX, EdgeEx, OBX | 225 | 5 sn | 20.000 |
  39 | 5 — Mutasyon | TWORS, ReverseSeq, THRORS, CentreInv, THROAS | 225 | 5 sn | 20.000 |
  40 | 6 — Final | En iyi konfigürasyonla 60 sn çalışma | 45 | 60 sn | 100.000 |
  41 | **Toplam** | | **3375** | | |
  42
  43 Her run hem süre hem iterasyon limitine tabidir; hangisine önce ulaşılırsa durur (`stop_reason`: `TIME_LIMIT` /
     `ITERATION_LIMIT` / `BOTH_LIMITS`).
  44
  45 ---
  46
  47 ## Çıktılar
  48
  47 ## Çıktılar
  48
  49 - **Excel (17 sayfa):** Ham runlar, aşama sıralamaları, final sonuçlar, görsel indeksi, doğrulama özeti
  50 - **Görseller:** Her veri seti için en iyi tur rotası + convergence grafiği (toplam 22 görsel)
  51 - **JSON:** Her final run için en iyi tur dizisi ve convergence geçmişi
  52 - **Checkpoint:** Her 50 run'da bir kayıt — kesintide kaldığı yerden devam
  53
  54 ---
  55
  56 ## Kurulum ve Çalıştırma
  57
  58 ```bash
  59 pip install numpy pandas matplotlib openpyxl
  60 ```
  61
  62 Proje dizininde `dataSets/` klasörü içinde `.txt` formatında TSP veri setleri bulunmalıdır.
  63
  64 ```bash
  65 # Hızlı test (2 veri seti, 2 seed, kısa limitler)
  66 python v6_time_iteration_unified_runner.py --mode smoke --base-dir .
  67
  68 # Tam çalıştırma (3375 run)
  69 python v6_time_iteration_unified_runner.py --mode full --base-dir .
  70 ```
