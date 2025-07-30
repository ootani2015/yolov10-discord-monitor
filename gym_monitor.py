import cv2
import torch
from ultralytics import YOLO
import requests
import datetime
import time

# --- 設定項目 ---
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1399649072444473344/4wKy6WsVShNId3gXcd3jJd-_grSthdhZ_i9-lgC5iu2O_LfjsaMXW4I0jzvdSlSbEGn6"
POST_INTERVAL_SECONDS = 10 # 何秒おきにDiscordに通知するか (例: 30秒ごとに通知)
CONFIDENCE_THRESHOLD = 0.4 # YOLOv10の信頼度の閾値

# ★追加・変更点: 混雑レベルの閾値設定
# ジムのキャパシティに合わせて、これらの数値を調整してください。
# 例: 0人 - 2人: 空いています, 3人 - 5人: やや混雑, 6人以上: 非常に混雑
THRESHOLD_EMPTY = 7   # この人数以下なら「空いています」
THRESHOLD_SLIGHTLY_CROWDED = 15 # この人数以下なら「やや混雑」、これを超えると「非常に混雑」

# --- YOLOv10モデルのロード ---
try:
    print("YOLOv10モデルをロードしています...")
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用するデバイス: {device}")
    model = YOLO('yolov10n.pt').to(device)
    print("モデルのロードが完了しました。")
except Exception as e:
    print(f"モデルのロード中にエラーが発生しました: {e}")
    exit(f"YOLOv10モデルのロードに失敗しました。インターネット接続を確認し、再度実行してください。エラー詳細: {e}")

# --- Webカメラの準備 ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("エラー: Webカメラにアクセスできません。カメラが接続されているか、他のアプリケーションで使用されていないか確認してください。")
    exit()

print("Webカメラに接続しました。")

last_post_time = time.time() # 最後の通知送信時刻を記録

# --- メインループ ---
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("エラー: フレームを読み込めませんでした。")
            break

        results = model(frame, conf=CONFIDENCE_THRESHOLD, classes=0, verbose=False)

        person_count = 0
        if results and results[0].boxes:
            person_count = len(results[0].boxes)

        print(f"現在時刻: {datetime.datetime.now().strftime('%H:%M:%S')} - ジム内の人数: {person_count}人")

        # 一定時間ごとにDiscordに通知を送信
        if time.time() - last_post_time >= POST_INTERVAL_SECONDS:
            # ★変更点: 混雑レベルに応じたメッセージの生成
            congestion_level = ""
            if person_count <= THRESHOLD_EMPTY:
                congestion_level = "空いています😊"
            elif person_count <= THRESHOLD_SLIGHTLY_CROWDED:
                congestion_level = "やや混雑しています🤔"
            else:
                congestion_level = "非常に混雑しています🚨"

            message_text = f"ジムの混雑状況をお知らせします。\n現在の人数: {person_count}人 ({congestion_level})"
            
            # Discord Webhook用のペイロード
            payload = {
                "content": message_text
            }
            headers = {"Content-Type": "application/json"}

            try:
                response = requests.post(DISCORD_WEBHOOK_URL, json=payload, headers=headers, timeout=5)
                response.raise_for_status()
                print(f"Discordに通知を送信しました。ステータスコード: {response.status_code}")
                last_post_time = time.time()

            except requests.exceptions.RequestException as e:
                print(f"Discord通知の送信に失敗しました: {e}")
            except Exception as e:
                print(f"Discord通知中に予期せぬエラーが発生しました: {e}")

        time.sleep(1)

except KeyboardInterrupt:
    print("スクリプトを終了します。")
except Exception as e:
    print(f"予期せぬエラーが発生し、スクリプトが停止しました: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()