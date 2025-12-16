import numpy as np
import torch
from PIL import Image

# --------- Config / Models ---------
MODEL_LIST = [
    "u2net",
    "u2netp",
    "u2net_human_seg",
    "u2net_cloth_seg",
    "silueta",
    "isnet-general-use",
    "isnet-anime",
    # Achtung: "sam" ist NICHT in allen rembg-builds/konfigurationen verfügbar
    # "sam",
]

def _ensure_rembg():
    """
    Import rembg late so ComfyUI doesn't fail to start if dependency is missing.
    """
    try:
        from rembg import new_session, remove
        return new_session, remove
    except Exception as e:
        raise RuntimeError(
            "rembg ist nicht installiert oder nicht importierbar.\n"
            "Installiere es im selben Python-Environment wie ComfyUI, z.B.:\n"
            "  pip install rembg\n\n"
            f"Original error: {repr(e)}"
        )

# --------- Conversions ---------
def tensor_to_pil(img: torch.Tensor) -> Image.Image:
    """
    ComfyUI IMAGE: torch float32 in [0..1], shape [B,H,W,C]
    """
    if img is None:
        raise ValueError("IMAGE ist None")
    if img.dim() != 4:
        raise ValueError(f"Unerwartete IMAGE shape: {tuple(img.shape)} (erwartet [B,H,W,C])")

    # Take first in batch
    arr = img[0].detach().cpu().numpy()
    arr = np.clip(arr, 0.0, 1.0)

    # Ensure 3 channels for PIL conversion
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.shape[-1] != 3:
        raise ValueError(f"Unerwartete Channel-Anzahl: {arr.shape[-1]} (erwartet 3 oder 4)")

    arr_u8 = (arr * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr_u8, mode="RGB")

def pil_rgba_to_image_and_mask(pil_img: Image.Image):
    """
    Returns:
      - IMAGE tensor [1,H,W,4] float32
      - MASK tensor  [1,H,W]   float32 (alpha)
    """
    rgba = pil_img.convert("RGBA")
    np_rgba = np.array(rgba).astype(np.float32) / 255.0  # [H,W,4]

    image = torch.from_numpy(np_rgba).unsqueeze(0)       # [1,H,W,4]
    mask = torch.from_numpy(np_rgba[..., 3]).unsqueeze(0) # [1,H,W]
    return image, mask

# --------- Node ---------
class ImageRemoveBackgroundRembg:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (MODEL_LIST,),
            },
            "optional": {
                # Kleine Qualitäts-Optionen: kann helfen bei "fransigen" Kanten
                "alpha_matting": ("BOOLEAN", {"default": False}),
                "alpha_matting_foreground_threshold": ("INT", {"default": 240, "min": 0, "max": 255}),
                "alpha_matting_background_threshold": ("INT", {"default": 10, "min": 0, "max": 255}),
                "alpha_matting_erode_size": ("INT", {"default": 10, "min": 0, "max": 100}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image_rgba", "mask_alpha")
    FUNCTION = "remove_background"
    CATEGORY = "image"

    def remove_background(
        self,
        image,
        model_name,
        alpha_matting=False,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=10,
    ):
        new_session, remove = _ensure_rembg()

        session = new_session(model_name)

        pil_in = tensor_to_pil(image)
        pil_out = remove(
            pil_in,
            session=session,
            alpha_matting=bool(alpha_matting),
            alpha_matting_foreground_threshold=int(alpha_matting_foreground_threshold),
            alpha_matting_background_threshold=int(alpha_matting_background_threshold),
            alpha_matting_erode_size=int(alpha_matting_erode_size),
        )

        out_image, out_mask = pil_rgba_to_image_and_mask(pil_out)
        return (out_image, out_mask)

# --------- Registration ---------
NODE_CLASS_MAPPINGS = {
    "ImageRemoveBackgroundRembg": ImageRemoveBackgroundRembg
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageRemoveBackgroundRembg": "Image Remove Background (rembg)"
}
