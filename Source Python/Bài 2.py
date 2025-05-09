    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import re
    import random

    # Đọc dữ liệu từ file CSV
    data = pd.read_csv('results.csv')

    # Thay thế "N/a" bằng NaN
    data.replace('N/a', np.nan, inplace=True)

    # Xác định các cột dự định là số, loại trừ các cột không liên quan
    intended_numeric_cols = [
        col for col in data.columns
        if col != 'stats_standard_Current age'
           and col not in ['Player', 'stats_standard_Nation', 'stats_standard_Squad', 'stats_standard_Position']
    ]

    # Chuyển đổi các cột dự định là số sang kiểu số, ép các lỗi thành NaN
    for col in intended_numeric_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Lọc để giữ lại các cột hoàn toàn là số
    numeric_cols = [col for col in intended_numeric_cols if data[col].dtype in ['float64', 'int64']]

    # Điền NaN bằng 0 cho các cột số
    data[numeric_cols] = data[numeric_cols].fillna(0)

    # Phân loại các chỉ số tấn công và phòng thủ
    attack_keywords = ['Goals', 'Assists', 'Shots', 'xG', 'xAG']
    defense_keywords = ['Tackles', 'Interceptions', 'Blocks', 'Clearances']
    attack_cols = [col for col in numeric_cols if any(keyword in col for keyword in attack_keywords)]
    defense_cols = [col for col in numeric_cols if any(keyword in col for keyword in defense_keywords)]

    # Chọn ngẫu nhiên 3 chỉ số từ mỗi loại (hoặc tất cả nếu ít hơn 3)
    selected_attack_cols = random.sample(attack_cols, min(3, len(attack_cols)))
    selected_defense_cols = random.sample(defense_cols, min(3, len(defense_cols)))

    # 1. Xác định top 3 cao nhất và thấp nhất cho mỗi chỉ số số
    with open('top_3.txt', 'w', encoding='utf-8') as f:
        for col in numeric_cols:
            f.write(f"Metric: {col}\n")
            f.write("Top 3 highest:\n")
            top_3 = data[['Player', col]].dropna().sort_values(by=col, ascending=False).head(3)
            for _, row in top_3.iterrows():
                f.write(f"- {row['Player']}: {row[col]:.2f}\n")
            f.write("Top 3 lowest:\n")
            bottom_3 = data[['Player', col]].dropna().sort_values(by=col).head(3)
            for _, row in bottom_3.iterrows():
                f.write(f"- {row['Player']}: {row[col]:.2f}\n")
            f.write("\n")

    # 2. Tính median, mean, và standard deviation cho tất cả cầu thủ
    all_stats = pd.DataFrame({
        '': ['all'],
        'Team': ['all']
    })
    for col in numeric_cols:
        all_stats[f'Median of {col}'] = [data[col].median()]
        all_stats[f'Mean of {col}'] = [data[col].mean()]
        all_stats[f'Std of {col}'] = [data[col].std()]

    # Tính toán theo từng đội
    team_stats = data.groupby('stats_standard_Squad')[numeric_cols].agg(['median', 'mean', 'std']).reset_index()

    # Làm phẳng MultiIndex cho các cột
    flattened_columns = [('Team', '')]
    for col in numeric_cols:
        flattened_columns.extend([(col, 'median'), (col, 'mean'), (col, 'std')])
    team_stats.columns = pd.MultiIndex.from_tuples(flattened_columns)

    # Đổi tên cột
    team_stats.columns = [f'Median of {col[0]}' if col[1] == 'median' else
                          f'Mean of {col[0]}' if col[1] == 'mean' else
                          f'Std of {col[0]}' if col[1] == 'std' else
                          'Team' for col in team_stats.columns]

    # Thêm cột chỉ số
    team_stats.insert(0, '', range(1, len(team_stats) + 1))

    # Đảm bảo các cột phù hợp và kết hợp
    common_columns = all_stats.columns.intersection(team_stats.columns)
    team_stats = team_stats.reindex(columns=common_columns, fill_value=np.nan)
    result_stats = pd.concat([all_stats, team_stats], ignore_index=True)
    result_stats.to_csv('results2.csv', index=False)

    # 3. Xác định đội có điểm số cao nhất cho mỗi chỉ số
    team_totals = data.groupby('stats_standard_Squad')[numeric_cols].sum()
    top_teams = team_totals.idxmax()

    print("Teams with the highest score for each metric:")
    for col, team in top_teams.items():
        print(f"- {col}: {team}")

    # Đánh giá đội thể hiện tốt nhất dựa trên các chỉ số chính
    key_stats = [
        'stats_standard_Goals', 'stats_standard_Assists',
        'stats_standard_xG: Expected Goals', 'stats_standard_xAG: Exp. Assisted Goals',
        'stats_keeper_Save Percentage'
    ]
    key_stats = [stat for stat in key_stats if stat in numeric_cols]
    team_means = data.groupby('stats_standard_Squad')[key_stats].mean()
    team_scores = team_means.sum(axis=1)
    best_team = team_scores.idxmax()
    best_score = team_scores.max()

    print(f"\nBest performing team: {best_team}")
    print(
        f"Reason: {best_team} leads with a total average score of {best_score:.2f} across key metrics (goals, assists, xG, xAG, Save%). "
        f"They demonstrate consistent offensive and defensive performance.")

    # 4. Vẽ biểu đồ histogram
    # Tạo thư mục histograms nếu chưa tồn tại
    if not os.path.exists('histograms'):
        os.makedirs('histograms')

    # Hàm làm sạch tên tệp
    def sanitize_filename(name):
        return re.sub(r'[^\w\-_\. ]', '_', name)

    # Vẽ biểu đồ cho tất cả cầu thủ
    plt.figure(figsize=(15, 10))
    plt.suptitle("Distribution of Selected Metrics for All Players", fontsize=16)
    for i, col in enumerate(selected_attack_cols, 1):
        plt.subplot(2, 3, i)
        plt.hist(data[col].dropna(), bins=20, color='blue', alpha=0.7)
        plt.title(f'Attack: {col}')
        plt.xlabel(col)
        plt.ylabel('Count')

    for i, col in enumerate(selected_defense_cols, 1):
        plt.subplot(2, 3, i + 3)
        plt.hist(data[col].dropna(), bins=20, color='green', alpha=0.7)
        plt.title(f'Defense: {col}')
        plt.xlabel(col)
        plt.ylabel('Count')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('histograms/hist_all_players.png')
    plt.show()

    # Vẽ biểu đồ cho từng đội
    teams = data['stats_standard_Squad'].unique()
    for team in teams:
        team_data = data[data['stats_standard_Squad'] == team]
        safe_team = sanitize_filename(team)

        plt.figure(figsize=(15, 10))
        plt.suptitle(f"Distribution of Selected Metrics for {team}", fontsize=16)
        for i, col in enumerate(selected_attack_cols, 1):
            plt.subplot(2, 3, i)
            plt.hist(team_data[col].dropna(), bins=20, color='blue', alpha=0.7)
            plt.title(f'Attack: {col}')
            plt.xlabel(col)
            plt.ylabel('Count')

        for i, col in enumerate(selected_defense_cols, 1):
            plt.subplot(2, 3, i + 3)
            plt.hist(team_data[col].dropna(), bins=20, color='green', alpha=0.7)
            plt.title(f'Defense: {col}')
            plt.xlabel(col)
            plt.ylabel('Count')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f'histograms/hist_{safe_team}.png')
        plt.show()