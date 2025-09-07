# src/step2.py
# YOLOモデルを使用して画像の物体検出を実行し、結果を表示する。

import cv2
from ultralytics import YOLO

# 画像を読み込む
image_path = "images/bus.jpg"
image = cv2.imread(image_path)

# 画像を読み込めなかった場合はエラーを表示して終了
if image is None:
	print("画像を読み込めませんでした。")
	exit()

# YOLOモデルを読み込む（事前学習済みモデルを使用）
model_path = 'models/yolo11n.pt'
model = YOLO(model_path)

# YOLOモデルを読み込めなかった場合はエラーを表示して終了
if model is None:
	print("YOLOモデルを読み込めませんでした。")
	exit()

# YOLO推論を実行
results = model(image)

# 推論結果を画像に描画
annotated_image = results[0].plot()

# 結果画像を表示
cv2.imshow("YOLO Detection Results", annotated_image)

# キーが押されるまで待つ
cv2.waitKey(0)
cv2.destroyAllWindows()
