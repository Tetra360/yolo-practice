# src/step6.py
# Webカメラの映像をリアルタイムでYOLO物体検出し、特定のラベルのみを描画して表示する

import cv2
from ultralytics import YOLO

# 検出対象のラベルを指定（例：人のみ）
target_class = 'person'

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

print(f"リアルタイム物体検出を開始します。'{target_class}'のみを検出します。'q'キーを押すと終了します。")

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

    # フレームをコピーして上書き用に準備
    annotated_frame = frame.copy()

    # 推論結果から特定のクラスのみを描画
    if results[0].boxes is not None:
        for box, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):
            # クラス名を取得
            class_name = model.names[int(cls)]
            if class_name == target_class:
                # xyxy形式のボックス座標を整数に変換
                x1, y1, x2, y2 = map(int, box)
                # 赤色でボックス描画 (BGR形式)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # ラベルを描画
                cv2.putText(annotated_frame, class_name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # 結果画像を表示
    cv2.imshow(f"Real-time YOLO Detection - {target_class} only", annotated_frame)

    # 'q'キーが押されたら終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# リソースを解放する
cap.release()
cv2.destroyAllWindows()
print("リアルタイム物体検出を終了しました。")
