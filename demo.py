from ultralytics import YOLO
import easyocr
import cv2


def main():
    img = cv2.imread("cam.jpg")
    model = YOLO("license_plate.pt")

    reader = easyocr.Reader(["en"], gpu=False)
    allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"

    results = model(img)[0]

    for res in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = res

        img_crop = img[int(y1) : int(y2), int(x1) : int(x2), :]

        img_gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
        bfilter = cv2.bilateralFilter(img_gray, 11, 17, 17)  # Noise reduction
        # edged = cv2.Canny(bfilter, 30, 200)  # Edge detection

        text_detections = reader.readtext(
            bfilter, paragraph=True, allowlist=allowlist, detail=0
        )

        cv2.imshow(text_detections[0], bfilter)

        cv2.waitKey(0)


if __name__ == "__main__":
    main()
