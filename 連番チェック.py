import os
import re
import glob

# フォルダーのパスを指定
folder_path = "photoaf"  # 適宜変更

# フォルダー内のPNGファイルを取得
files = glob.glob(os.path.join(folder_path, "frame_*.png"))

# ファイル名から番号を抽出
numbers = []
pattern = re.compile(r"frame_(\d{4})\.png")

for file in files:
    match = pattern.search(os.path.basename(file))
    if match:
        numbers.append(int(match.group(1)))

# 昇順にソート
numbers.sort()

# 連番でないものをチェック
missing_numbers = []
for i in range(1, len(numbers)):
    expected = numbers[i-1] + 1
    if numbers[i] != expected:
        missing_numbers.extend(range(expected, numbers[i]))

# 結果を表示
if missing_numbers:
    print("Missing frame numbers:", missing_numbers)
else:
    print("All frames are sequential.")
