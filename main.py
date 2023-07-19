import cv2

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

    while True:
        ret, frame = cap.read()

        result = model(frame)[0]
        detections = sv.Detections.from_yolov8(result)
        labels = []

        for license_plate in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            license_plate_crop = frame[int(y1) : int(y2), int(x1) : int(x2), :]

            # license_plate_crop_gray = cv2.cvtColor(
            #     license_plate_crop, cv2.COLOR_BGR2GRAY
            # )

            text_detections = reader.readtext(
                license_plate_crop, paragraph=True, allowlist=allowlist, detail=0
            )
            try:
                for text in text_detections:
                    print(text)
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
            except:
                labels.append("Cannot read the plate")

        frame = box_anotator.annotate(scene=frame, detections=detections, labels=labels)

        cv2.imshow("plate-cam", frame)

        if cv2.waitKey(30) == 27:
            break


if __name__ == "__main__":
    main()
