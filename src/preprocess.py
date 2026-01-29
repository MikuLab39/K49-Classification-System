import io
import torch
from PIL import Image, ImageOps
import torchvision.transforms as transforms
from .config import STATS_MEAN, STATS_STD

LABEL_MAP = {
    0: 'あ', 1: 'い', 2: 'う', 3: 'え', 4: 'お',
    5: 'か', 6: 'き', 7: 'く', 8: 'け', 9: 'こ',
    10: 'さ', 11: 'し', 12: 'す', 13: 'せ', 14: 'そ',
    15: 'た', 16: 'ち', 17: 'つ', 18: 'て', 19: 'と',
    20: 'な', 21: 'に', 22: 'ぬ', 23: 'ね', 24: 'の',
    25: 'は', 26: 'ひ', 27: 'ふ', 28: 'へ', 29: 'ほ',
    30: 'ま', 31: 'み', 32: 'む', 33: 'め', 34: 'も',
    35: 'や', 36: 'ゆ', 37: 'よ',
    38: 'ら', 39: 'り', 40: 'る', 41: 'れ', 42: 'ろ',
    43: 'わ', 44: 'ゐ', 45: 'ゑ', 46: 'を', 47: 'ん',
    48: 'ゝ'
}

#Preprocesses raw image bytes for inference.

def transform_image(image_bytes: bytes) -> torch.Tensor:

    image = Image.open(io.BytesIO(image_bytes)).convert('L')

    # Invert colors if the background is bright
    if image.getpixel((0, 0)) > 128:
        image = ImageOps.invert(image)

    # Pad to square to preserve aspect ratio before resizing.
    max_dim = max(image.size)
    new_image = Image.new('L', (max_dim, max_dim), 0)  # 0 = Black background
    
    # Center the original image on the new canvas
    offset_x = (max_dim - image.width) // 2
    offset_y = (max_dim - image.height) // 2
    new_image.paste(image, (offset_x, offset_y))
    image = new_image

    # Standard tensor transformation
    pipeline = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(STATS_MEAN, STATS_STD)
    ])
    
    # Add batch dimension
    return pipeline(image).unsqueeze(0)

def get_character(index: int) -> str:
    return LABEL_MAP.get(index, "?")
