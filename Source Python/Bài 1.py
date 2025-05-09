    # =========================
    # 1. CÀI ĐẶT TRÊN COLAB / MÁY LOCAL
    # =========================
    !pip install selenium
    !apt-get update -y
    !apt install -y chromium-chromedriver
    !cp /usr/lib/chromium-browser/chromedriver /usr/bin
    
    # =========================
    # 2. IMPORT LIBRARY
    # =========================
    import time
    import pandas as pd
    import re
    from bs4 import BeautifulSoup
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    
    # =========================
    # 3. CẤU HÌNH SELENIUM CHROME
    # =========================
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(options=options)
    
    # =========================
    # 4. ĐỊNH NGHĨA CÁC BẢNG CẦN LẤY
    # =========================
    tables = [
        {
            "url": "https://fbref.com/en/comps/9/stats/Premier-League-Stats",
            "id": "stats_standard",
            "cols": ["Nation", "Squad", "Position", "Current age", "Matches Played", "Minutes",
                     "Goals", "Assists", "Yellow Cards", "Red Cards",
                     "xG: Expected Goals", "xAG: Exp. Assisted Goals",
                     "Progressive Carries", "Progressive Passes", "Progressive Passes Rec",
                     "Goals/90", "Assists/90", "xG/90", "xAG/90"]
        },
        {
            "url": "https://fbref.com/en/comps/9/keepers/Premier-League-Stats",
            "id": "stats_keeper",
            "cols": ["Goals Against/90", "Save Percentage", "Clean Sheet Percentage", "Save% (Penalty Kicks)"]
        },
        {
            "url": "https://fbref.com/en/comps/9/shooting/Premier-League-Stats",
            "id": "stats_shooting",
            "cols": ["Shots on Target %", "Shots on target/90", "Goals/Shot", "Average Shot Distance"]
        },
        {
            "url": "https://fbref.com/en/comps/9/passing/Premier-League-Stats",
            "id": "stats_passing",
            "cols": ["Passes Completed", "Pass Completion %", "Total Passing Distance",
                     "Passes Completed (Short)", "Pass Completion % (Medium)", "Pass Completion % (Long)",
                     "Key Passes", "Passes into Final Third", "Passes into Penalty Area",
                     "Crosses into Penalty Area", "Progressive Passes"]
        },
        {
            "url": "https://fbref.com/en/comps/9/gca/Premier-League-Stats",
            "id": "stats_gca",
            "cols": ["Shot-Creating Actions", "Shot-Creating Actions/90",
                     "Goal-Creating Actions", "Goal-Creating Actions/90"]
        },
        {
            "url": "https://fbref.com/en/comps/9/defense/Premier-League-Stats",
            "id": "stats_defense",
            "cols": ["Tackles", "Tackles Won", "Dribbles Challenged", "Challenges Lost",
                     "Blocks", "Shots Blocked", "Passes Blocked", "Interceptions"]
        },
        {
            "url": "https://fbref.com/en/comps/9/possession/Premier-League-Stats",
            "id": "stats_possession",
            "cols": ["Touches", "Touches (Def Pen)", "Touches (Def 3rd)", "Touches (Mid 3rd)",
                     "Touches (Att 3rd)", "Touches (Att Pen)", "Take-Ons Attempted",
                     "Successful Take-On %", "Tackled During Take-On Percentage",
                     "Carries", "Progressive Carrying Distance", "Progressive Carries",
                     "Carries into Final Third", "Carries into Penalty Area",
                     "Miscontrols", "Dispossessed", "Passes Received", "Progressive Passes Rec"]
        },
        {
            "url": "https://fbref.com/en/comps/9/misc/Premier-League-Stats",
            "id": "stats_misc",
            "cols": ["Yellow Cards", "Red Cards", "Fouls Committed", "Fouls Drawn",
                     "Offsides", "Crosses", "Ball Recoveries", "Aerials Won",
                     "Aerials Lost", "% of Aerials Won"]
        }
    ]
    
    # =========================
    # 5. HÀM SCRAPE 1 BẢNG
    # =========================
    def scrape_table(url, table_id, required_cols):
        print(f"📥 Đang tải bảng {table_id} …")
        driver.get(url)
        time.sleep(3)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        tbl = soup.find('table', {'id': table_id})
        if tbl is None:
            print(f"❌ Bảng {table_id} không tìm thấy!")
            return None
    
        # Header
        header_cells = tbl.find('thead').find_all('tr')[-1].find_all('th')
        headers = []
        for th in header_cells:
            labels = th.get('aria-label') or th.text.strip()
            headers.append(labels)
        headers[0] = 'Player'
    
        # Data
        rows = []
        for tr in tbl.find('tbody').find_all('tr'):
            if tr.get('class') == ['thead']:
                continue
            cols = tr.find_all(['th','td'])
            if len(cols) != len(headers):
                continue
            vals = []
            for i, cell in enumerate(cols):
                txt = cell.get_text(strip=True)
                if i == 0:
                    txt = re.sub(r'^[\d\.\s]+', '', re.sub(r'[(),]', '', txt)).strip()
                vals.append(txt)
            rows.append(vals)
    
        df = pd.DataFrame(rows, columns=headers).set_index('Player')
        df = df.add_prefix(f"{table_id}_")
    
        # Lấy các cột cần
        want = [f"{table_id}_{c}" for c in required_cols]
        have = [c for c in want if c in df.columns]
        missing = set(want) - set(have)
        if missing:
            print(f"⚠️ Bảng {table_id} thiếu cột: {sorted(missing)}")
        return df[have]
    
    # =========================
    # 6. SCRAPE TOÀN BỘ
    # =========================
    dfs = {}
    for t in tables:
        df = scrape_table(t["url"], t["id"], t["cols"])
        dfs[t["id"]] = df
    
    driver.quit()
    
    # =========================
    # 7. GỘP DỮ LIỆU
    # =========================
    std = dfs['stats_standard']
    keepers = dfs['stats_keeper']
    others = [df for key, df in dfs.items() if key not in ['stats_standard','stats_keeper']]
    
    merged = std.copy()
    for o in others:
        merged = merged.join(o, how='left')
    
    # GK: merge keeper stats cho thủ môn
    if keepers is not None:
        is_gk = merged['stats_standard_Position'].str.contains('GK', na=False)
        for col in keepers.columns:
            merged[col] = pd.NA
            merged.loc[is_gk, col] = keepers.reindex(merged.index).loc[is_gk, col]
    
    # =========================
    # 8. CHỈNH LẠI CỘT PLAYER
    # =========================
    merged = merged.reset_index()
    merged['Player'] = merged['Player'].apply(lambda x: x[1] if isinstance(x, tuple) else x)
    player_col = merged.pop('Player')
    merged.insert(0, 'Player', player_col)
    
    # =========================
    # 9. FILTER >90 PHÚT
    # =========================
    merged['stats_standard_Minutes'] = (
        merged['stats_standard_Minutes']
        .str.replace(',', '', regex=False)
        .astype(float, errors='ignore')
    )
    merged = merged[ merged['stats_standard_Minutes'] > 90 ]
    
    # =========================
    # 10. XOÁ TRÙNG PLAYER
    # =========================
    merged = merged.drop_duplicates(subset=['Player'])
    
    # =========================
    # 11. THAY CÁC GIÁ TRỊ RỖNG BẰNG 'N/a' (NHƯNG GIỮ NGUYÊN 0)
    # =========================
    merged = merged.replace(['', '–'], pd.NA)
    merged = merged.fillna('N/a')
    
    # =========================
        # 12. SẮP XẾP VÀ LƯU CSV
        # =========================
    merged = merged.sort_values('Player')
    print(f"✅ Tổng số cầu thủ cuối cùng: {len(merged)}")
    merged.to_csv('results.csv', index=False)
    print("📄 Đã lưu kết quả vào results.csv")