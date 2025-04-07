import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

# Canny法でエッジ検出
import numpy as np
import cv2
import os

def canny(img_path: str):
    print('hallt')
    np.set_printoptions(threshold=np.inf)
    plt.gray()

    # 画像の読み込み（グレースケール）
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    height, width = img.shape[:2]  # 画像のサイズ取得

    # エッジ検出（Canny法）
    edges = cv2.Canny(img, 50, 100)

    # エッジポイントの座標を取得
    edge_points = np.column_stack(np.where(edges > 0))  # (y, x) の順

    # x, y に変換（左下原点にする）
    edge_points = edge_points[:, ::-1]  # (y, x) -> (x, y)
    edge_points[:, 1] = height - edge_points[:, 1]  # y軸を反転

    # ランダムに 20% の点を選択
    num_rows = int(edge_points.shape[0] * 0.2)

    # サンプル数が0未満にならないように制限
    num_rows = max(1, num_rows)  # 最低でも1点は選ぶ

    # サンプル数が元のデータ数を超えないようにする
    num_rows = min(num_rows, edge_points.shape[0])

    if edge_points.shape[0] > 0:
        selected_indices = np.random.choice(edge_points.shape[0], num_rows, replace=False)
        selected_array = edge_points[selected_indices]
        return selected_array
    else:
        print("エッジが検出されませんでした。")
        return None


# 最も近い点を順に選んで一筆書き
def selse(selected_array):
    current_point = selected_array[np.random.choice(len(selected_array))]
    visited = [current_point]
    remaining_points = selected_array.tolist()

    while remaining_points:
        distances = np.linalg.norm(np.array(remaining_points) - current_point, axis=1)
        min_index = np.argmin(distances)
        closest_point = remaining_points.pop(min_index)
        
        visited.append(closest_point)
        current_point = closest_point

    visited_points = np.array(visited)
    return visited_points

# 一筆書きの点にフーリエ変換を適用
def furi(visited_points):
    X = np.fft.fft(visited_points[:, 0])
    Y = np.fft.fft(visited_points[:, 1])

    N = X.shape[0]
    nlim = min(150, N//2)  # nlimをXの長さの半分以下に制限

    s = np.arange(0, N, 1.0)
    x_data = X[0] * np.cos(2 * np.pi * 0.0 / N * s) / N
    y_data = Y[0] * np.cos(2 * np.pi * 0.0 / N * s) / N

    for n in range(1, nlim + 1):
        xan = (X[n] - X[N - n]) * (0.0 + 1.0j)
        xbn = (X[n] + X[N - n])
        yan = (Y[n] - Y[N - n]) * (0.0 + 1.0j)
        ybn = (Y[n] + Y[N - n])
        x_data += xan * np.sin(2 * np.pi * n / N * s) / N + xbn * np.cos(2 * np.pi * n / N * s) / N
        y_data += yan * np.sin(2 * np.pi * n / N * s) / N + ybn * np.cos(2 * np.pi * n / N * s) / N

    return np.real(x_data), np.real(y_data)


# フォルダ内の全ての画像に処理を行い、新しいフォルダに保存する関数
def process_and_save_images(input_folder: str, output_folder: str):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, filename)

            selected_array = canny(img_path)
            if selected_array is not None:
                visited_points = selse(selected_array)
                x_data, y_data = furi(visited_points)

                # 新しいファイル名を作成
                output_path = os.path.join(output_folder, filename)

                # フーリエ変換の結果をプロットして保存
                '''
                plt.xlim(0, 1600) 
                plt.ylim(0, 1400)    
                plt.xticks([400,800,1200,1600]) 
                plt.xticklabels(["A", "B", "C", "D", "E"])
                plt.figure(figsize=(8, 8))
                '''
                plt.xticks([])
                plt.yticks([])
                plt.plot(x_data, y_data, color='black')
                plt.axis("equal")
                plt.savefig(output_path, dpi=300)
                plt.close()



# 実行例
input_folder = "beforetrim"  # 入力画像フォルダ
output_folder = "photoaf"  # 出力画像フォルダ

process_and_save_images(input_folder, output_folder)













