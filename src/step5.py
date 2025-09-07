# src/step5.py
# Webカメラの映像をリアルタイムでYOLO物体検出し、結果を描画して表示する

import cv2
from ultralytics import YOLO

# YOLOモデルを読み込む（事前学習済みモデルを使用）
model_path = 'models/yolo11n.pt'
model = YOLO(model_path)

# YOLOモデルを読み込めなかった場合はエラーを表示して終了
if model is None:
    print("YOLOモデルを読み込めませんでした。")
    exit()

# Webカメラを開く
cap = cv2.VideoCapture(0)

# カメラが開けない場合は終了
if not cap.isOpened():
    print("カメラを開けませんでした")
    exit()

print("リアルタイム物体検出を開始します。'q'キーを押すと終了します。")

# メインループ
while True:
    # フレームを読み込む
    ret, frame = cap.read()

    # フレームが読み込めない場合は終了
    if not ret:
        print("フレームを読み込めませんでした")
        break

    # YOLO推論を実行
    results = model(frame)

    # 推論結果を画像に描画
    annotated_frame = results[0].plot()

    # 結果画像を表示
    cv2.imshow("Real-time YOLO Detection", annotated_frame)

    # 'q'キーが押されたら終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# リソースを解放する
cap.release()
cv2.destroyAllWindows()
print("リアルタイム物体検出を終了しました。")
