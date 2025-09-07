# src/step3.py
# YOLOモデルを使用して画像の物体検出を実行し、結果を表示 さらに、バスのみを抽出して表示する。

import cv2
from ultralytics import YOLO

# 画像を読み込む
image_path = "images/bus.jpg"
image = cv2.imread(image_path)

if image is None:
    print("画像を読み込めませんでした。")
    exit()

# YOLOモデルを読み込む
model_path = 'models/yolo11n.pt'
model = YOLO(model_path)

if model is None:
    print("YOLOモデルを読み込めませんでした。")
    exit()

# YOLO推論
results = model(image)

# 推論結果を描画
# imageをコピーして上書き
annotated_image = image.copy()

for box, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):
    # クラス名を取得
    class_name = model.names[int(cls)]
    if class_name == 'bus':
        # xyxy形式のボックス座標を整数に変換
        x1, y1, x2, y2 = map(int, box)
        # 赤色でボックス描画 (BGR形式)
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # ラベルを描画
        cv2.putText(annotated_image, class_name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

# 結果画像を表示
cv2.imshow("YOLO Detection Results", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
