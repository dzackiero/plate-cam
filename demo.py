from ultralytics import YOLO
import easyocr
import cv2


def main():
    img = cv2.imread("cam.jpg")
    model = YOLO("license_plate.pt")
    results = model.predict(img)

    for res in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = res

        img_crop = img[int(y1) : int(y2), int(x1) : int(x2), :]

        cv2.imshow("crop", img_crop)


if __name__ == "__main__":
    main()
