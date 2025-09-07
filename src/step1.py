# src/step1.py
# cv2で画像を読み込んで表示する

import cv2

# 画像を読み込む
image = cv2.imread("images/bus.jpg")

# 画像を表示する
cv2.imshow("Image", image)

# キーが押されるまで待つ
cv2.waitKey(0)
cv2.destroyAllWindows()