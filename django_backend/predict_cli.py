"""
Terminal-first OCR prediction helper for sentence-level models.
"""

import argparse

import cv2
import torch
from PIL import Image as PILImage
from torchvision import transforms

from char_map import char_to_idx, idx_to_char
from model import CRNN
from post_processing import correct_sentence


def parse_args():
    parser = argparse.ArgumentParser(description="Run OCR prediction from the terminal")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model_sentences.pth",
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Resize width for the OCR model",
    )
    parser.add_argument(
        "--disable-post-process",
        action="store_true",
        help="Skip spell correction post-processing",
    )
    return parser.parse_args()


def decode_prediction(preds, pred_sizes):
    decoded_texts = []
    preds = preds.permute(1, 0, 2).cpu()
    preds = torch.argmax(preds, dim=2)

    for i in range(preds.size(0)):
        pred_seq = preds[i][: pred_sizes[i]]
        prev_char = -1
        text = ""
        for idx in pred_seq:
            idx = idx.item()
            if idx != prev_char and idx != len(char_to_idx):
                text += idx_to_char.get(idx, "")
            prev_char = idx
        decoded_texts.append(text)
    return decoded_texts


def load_model(checkpoint_path, device):
    model = CRNN(img_height=32, nn_classes=len(char_to_idx) + 1)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict(image_path, checkpoint_path, width=512, use_post_process=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, device)

    transform = transforms.Compose(
        [
            transforms.Resize((32, width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    pil_image = PILImage.fromarray(image)
    image_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        outputs = torch.log_softmax(outputs, 2)
        pred_sizes = torch.full(
            size=(outputs.size(1),),
            fill_value=outputs.size(0),
            dtype=torch.int32,
        )
        decoded = decode_prediction(outputs, pred_sizes)

    text = decoded[0]
    if use_post_process:
        text = correct_sentence(text)
    return text


def main():
    args = parse_args()
    text = predict(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        width=args.width,
        use_post_process=not args.disable_post_process,
    )
    print(text)


if __name__ == "__main__":
    main()
