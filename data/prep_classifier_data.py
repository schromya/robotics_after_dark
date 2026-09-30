"""Crop and label camera images for torchvision ImageFolder.
Source: ChatGPT.

Usage:
python data/prep_classifier_data.py --input data/top_camera/20260930_020532_197587/ --output data/classifier --crop 200 200 640 360

Crop: X Y WIDTH HEIGHT in source pixels.

Controls: 1=success, 0=failure, s=skip, q=quit.
"""

import argparse
from pathlib import Path

import cv2


EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
WINDOW = "Label image: 1=success | 0=failure | s=skip | q=quit"


def load_image(path):
    """
    Load an image from disk.
    Args:
        path: Input image path.
    Returns:
        The image array in OpenCV's BGR format.
    """
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def crop_image(image, crop):
    """
    Crop an image without resizing it.
    Args:
        image: Input image array.
        crop: [x, y, width, height] rectangle in source pixels.
    Returns:
        The cropped image array.
    """
    x, y, width, height = crop
    if min(x, y) < 0 or min(width, height) <= 0:
        raise ValueError("Crop coordinates must be nonnegative and dimensions positive.")
    if x + width > image.shape[1] or y + height > image.shape[0]:
        raise ValueError("Crop extends outside the image.")
    return image[y:y + height, x:x + width]


def label_image(image):
    """
    Display an image and wait for a keyboard label.
    Args:
        image: Cropped image array to display.
    Returns:
        "success", "failure", "skip", or "quit".
    """
    cv2.imshow(WINDOW, image)
    labels = {ord("1"): "success", ord("0"): "failure", ord("s"): "skip", ord("q"): "quit"}
    while True:
        key = cv2.waitKey(100) & 0xFF
        if key == 27 or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            return "quit"
        if key in labels:
            return labels[key]


def save_image(image, output, filename, label):
    """
    Save a labeled image as a PNG.
    Args:
        image: Cropped BGR image array.
        output: Dataset root containing success and failure directories.
        filename: Output PNG filename.
        label: "success" or "failure".
    Returns:
        Path to the saved image.
    """
    for folder in ("success", "failure"):
        if (output / folder / filename).exists():
            raise FileExistsError(f"Image already labeled: {filename}. Use a fresh output folder.")
    path = output / label / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Failed to save image: {path}")
    return path


def prepare_data(source, output, crop):
    """
    Load, crop, label, and save each image in a directory.
    Args:
        source: Directory of original images.
        output: Destination dataset directory.
        crop: [x, y, width, height] rectangle applied to every image.
    """
    files = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not files:
        raise ValueError(f"No images found in {source}")
    for label in ("success", "failure"):
        (output / label).mkdir(parents=True, exist_ok=True)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    try:
        for path in files:
            print(path.name)
            image = crop_image(load_image(path), crop)
            label = label_image(image)
            if label == "quit":
                break
            if label == "skip":
                continue
            filename = f"{source.resolve().name}_{path.stem}.png"
            print(f"Saved {save_image(image, output, filename, label)}")
    finally:
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/classifier"))
    parser.add_argument("--crop", type=int, nargs=4, required=True, metavar=("X", "Y", "WIDTH", "HEIGHT"))
    args = parser.parse_args()
    prepare_data(args.input, args.output, args.crop)


if __name__ == "__main__":
    main()
