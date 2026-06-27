import helper
import torch
from net import ResNetEncoder
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Inference AdaFace ResNet Encoder")

    # Path
    parser.add_argument("--img_path1", type=str)
    parser.add_argument("--img_path2", type=str)
    parser.add_argument("--checkpoint_path", type=str, default=r"D:\private\face_recognition\embedded_model\checkpoints\best.pth")
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--threshold", default=0.6, help="Mức threshold mong muốn", type=float)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    img_path1 = args.img_path1
    img_path2 = args.img_path2
    checkpoint_path = args.checkpoint_path
    threshold = args.threshold

    model = ResNetEncoder(embedding_dim=args.embedding_dim)
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict["model"])
    print("Load model từ checkpoint thành công")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model is inferring on {device}.")

    cosine = helper.face_verification(img_path1=img_path1, img_path2=img_path2, device=device, model=model)
    cosine = cosine.item()
    if cosine >= threshold:
        print(f"Hai ảnh cùng người với điểm số {cosine:.4f} và threshold là {threshold}")
    else:
        print(f"Hai ảnh khác nhau với điểm số {cosine:.4f} và threshold là {threshold}")

