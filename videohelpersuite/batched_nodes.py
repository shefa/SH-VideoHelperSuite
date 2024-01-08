import torch
from nodes import VAEEncode
from .logger import logger

class VAEDecodeBatched:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT", ),
                "vae": ("VAE", ),
                "per_batch": ("INT", {"default": 16, "min": 1})
                }
            }
    
    CATEGORY = "SH Video Helper Suite 🎥🅥🅗🅢/batched nodes"

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"

    def decode(self, vae, samples, per_batch):
        decoded = []
        logger.info("starting decode")
        for start_idx in range(0, samples["samples"].shape[0], per_batch):
            logger.info(str(start_idx)+"/"+str(samples["samples"].shape[0]))
            decoded.append(vae.decode(samples["samples"][start_idx:start_idx+per_batch]))
        logger.info("decode ended!")
        ret = torch.cat(decoded, dim=0)
        logger.info("really ended!")
        return (ret, )


class VAEEncodeBatched:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pixels": ("IMAGE", ), "vae": ("VAE", ),
                "per_batch": ("INT", {"default": 16, "min": 1})
                }
            }
    
    CATEGORY = "SH Video Helper Suite 🎥🅥🅗🅢/batched nodes"

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"

    def encode(self, vae, pixels, per_batch):
        t = []
        for start_idx in range(0, pixels.shape[0], per_batch):
            sub_pixels = VAEEncode.vae_encode_crop_pixels(pixels[start_idx:start_idx+per_batch])
            t.append(vae.encode(sub_pixels[:,:,:,:3]))
        return ({"samples": torch.cat(t, dim=0)}, )
