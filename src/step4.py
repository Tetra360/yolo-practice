# src/step4.py
# webカメラの映像をウィンドウ表示する

import cv2

# webカメラを開く
cap = cv2.VideoCapture(0)

# カメラが開けない場合は終了
if not cap.isOpened():
    print("カメラを開けませんでした")
    exit()

# メインループ
while True:
    # フレームを読み込む
    ret, frame = cap.read()

    # フレームが読み込めない場合は終了
    if not ret:
        print("フレームを読み込めませんでした")
        break

    # フレームを表示する
    cv2.imshow("Web Camera", frame)

    # 'q'キーが押されたら終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# リソースを解放する
cap.release()
cv2.destroyAllWindows()
