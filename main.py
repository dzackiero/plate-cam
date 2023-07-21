import cv2
import csv
import re
from datetime import datetime

from ultralytics import YOLO
import supervision as sv
import easyocr


def main():
    cap = cv2.VideoCapture(0)

    # making YOLO instance using license_plate model
    model = YOLO("license_plate.pt")
    box_anotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1,
    )

    reader = easyocr.Reader(["en"], gpu=False)
    allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
    regex = "([A-Z]{1,3})(\d{1,4})([A-Z]{0,3})$"

    fieldnames = ["license-plate", "score", "datetime"]
    listed_license = []
    datas = []

    while True:
        ret, frame = cap.read()

        result = model(frame)[0]
        detections = sv.Detections.from_yolov8(result)
        labels = []

        for license_plate in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            license_plate_crop = frame[int(y1) : int(y2), int(x1) : int(x2), :]

            license_plate_crop_gray = cv2.cvtColor(
                license_plate_crop, cv2.COLOR_BGR2GRAY
            )
            bfilter = cv2.bilateralFilter(license_plate_crop_gray, 11, 17, 17)
            # edged = cv2.Canny(bfilter, 30, 200)

            text_detections = reader.readtext(
                bfilter, paragraph=True, allowlist=allowlist, detail=0
            )
            try:
                for text in text_detections:
                    text = text.replace(" ", "")
                    count = 0
                    for i, c in enumerate(text[::-1]):
                        if c.isdigit():
                            count += 1
                        elif c.isalpha() and text[::-1][i + 1].isdigit():
                            count += 1
                        else:
                            break
                    labels.append(f"{text[:-count]} {score:0.2f}")

                    if re.match(regex, text[:-count]):
                        if text[:-count] not in listed_license:
                            listed_license.append(text[:-count])
                            datas.append(
                                {
                                    "license-plate": text[:-count],
                                    "score": score,
                                    "datetime": datetime.now().strftime(
                                        "%d/%m/%Y, %H:%M:%S"
                                    ),
                                }
                            )

            except:
                labels.append("Cannot read the plate")

        frame = box_anotator.annotate(scene=frame, detections=detections, labels=labels)

        cv2.imshow("plate-cam", frame)

        if cv2.waitKey(30) == 27:
            with open("datas.csv", "w", encoding="UTF8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(datas)
            break


if __name__ == "__main__":
    main()
