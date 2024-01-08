import os
import sys
import json
import subprocess
import numpy as np
import re
from typing import List
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from pathlib import Path

import folder_paths
from .logger import logger
from .image_latent_nodes import DuplicateImages, DuplicateLatents, GetImageCount, GetLatentCount, MergeImages, MergeLatents, SelectEveryNthImage, SelectEveryNthLatent, SplitLatents, SplitImages
from .load_video_nodes import LoadVideoUpload, LoadVideoPath
from .load_images_nodes import LoadImagesFromDirectoryUpload, LoadImagesFromDirectoryPath
from .batched_nodes import VAEEncodeBatched, VAEDecodeBatched
from .utils import ffmpeg_path, get_audio, hash_path, validate_path

folder_paths.folder_names_and_paths["VHS_video_formats"] = (
    [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "video_formats"),
    ],
    [".json"]
)

def gen_format_widgets(video_format):
    for k in video_format:
        if k.endswith("_pass"):
            for i in range(len(video_format[k])):
                if isinstance(video_format[k][i], list):
                    item = [video_format[k][i]]
                    yield item
                    video_format[k][i] = item[0]
        else:
            if isinstance(video_format[k], list):
                item = [video_format[k]]
                yield item
                video_format[k] = item[0]

def get_video_formats():
    formats = []
    for format_name in folder_paths.get_filename_list("VHS_video_formats"):
        format_name = format_name[:-5]
        video_format_path = folder_paths.get_full_path("VHS_video_formats", format_name + ".json")
        with open(video_format_path, 'r') as stream:
            video_format = json.load(stream)
        widgets = [w[0] for w in gen_format_widgets(video_format)]
        if (len(widgets) > 0):
            formats.append(["video/" + format_name, widgets])
        else:
            formats.append("video/" + format_name)
    return formats

def apply_format_widgets(format_name, kwargs):
    video_format_path = folder_paths.get_full_path("VHS_video_formats", format_name + ".json")
    with open(video_format_path, 'r') as stream:
        video_format = json.load(stream)
    for w in gen_format_widgets(video_format):
        assert(w[0][0] in kwargs)
        w[0] = str(kwargs[w[0][0]])
    return video_format

def tensor_to_int(tensor, bits):
    #TODO: investigate benefit of rounding by adding 0.5 before clip/cast
    tensor = tensor.cpu().numpy() * (2**bits-1)
    return np.clip(tensor, 0, (2**bits-1))
def tensor_to_shorts(tensor):
    return tensor_to_int(tensor, 16).astype(np.uint16)
def tensor_to_bytes(tensor):
    return tensor_to_int(tensor, 8).astype(np.uint8)

class VideoCombine:
    @classmethod
    def INPUT_TYPES(s):
        #Hide ffmpeg formats if ffmpeg isn't available
        if ffmpeg_path is not None:
            ffmpeg_formats = get_video_formats()
        else:
            ffmpeg_formats = []
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_rate": (
                    "INT",
                    {"default": 8, "min": 1, "step": 1},
                ),
                "loop_count": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "filename_prefix": ("STRING", {"default": "AnimateDiff"}),
                "format": (["image/gif", "image/webp"] + ffmpeg_formats,),
                "pingpong": ("BOOLEAN", {"default": False}),
                "save_output": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "audio": ("VHS_AUDIO",),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID"
            },
        }

    RETURN_TYPES = ("VHS_FILENAMES",)
    RETURN_NAMES = ("Filenames",)
    OUTPUT_NODE = True
    CATEGORY = "SH Video Helper Suite 🎥🅥🅗🅢"
    FUNCTION = "combine_video"

    def combine_video(
        self,
        images,
        frame_rate: int,
        loop_count: int,
        filename_prefix="AnimateDiff",
        format="image/gif",
        pingpong=False,
        save_output=True,
        prompt=None,
        extra_pnginfo=None,
        audio=None,
        unique_id=None,
    ):
        logger.info('combine_video_init')
        kwargs = prompt[unique_id]['inputs']
        # convert images to numpy
        logger.info('1')
        # get output information
        output_dir = (
            folder_paths.get_output_directory()
            if save_output
            else folder_paths.get_temp_directory()
        )
        logger.info('2')
        logger.info(output_dir)
        
        (
            full_output_folder,
            filename,
            _,
            subfolder,
            _,
        ) = folder_paths.get_save_image_path(filename_prefix, output_dir)
        output_files = []
        logger.info('3')
        
        metadata = PngInfo()
        logger.info('4')
        video_metadata = {}
        if prompt is not None:
            metadata.add_text("prompt", json.dumps(prompt))
            video_metadata["prompt"] = prompt
            logger.info('5')
        if extra_pnginfo is not None:
            logger.info('6')
            for x in extra_pnginfo:
                logger.info('7')
                metadata.add_text(x, json.dumps(extra_pnginfo[x]))
                video_metadata[x] = extra_pnginfo[x]
            logger.info('8')

        logger.info('9')
        # comfy counter workaround
        max_counter = 0
        logger.info('10...')
        # Loop through the existing files
        matcher = re.compile(f"{re.escape(filename)}_(\d+)\D*\.[a-zA-Z0-9]+")
        for existing_file in os.listdir(full_output_folder):
            # Check if the file matches the expected format
            match = matcher.fullmatch(existing_file)
            if match:
                # Extract the numeric portion of the filename
                file_counter = int(match.group(1))
                # Update the maximum counter value if necessary
                if file_counter > max_counter:
                    max_counter = file_counter
        logger.info('11')
        # Increment the counter by 1 to get the next available value
        counter = max_counter + 1
        logger.info('12')
        # save first frame as png to keep metadata
        file = f"{filename}_{counter:05}.png"
        logger.info(file)
        file_path = os.path.join(full_output_folder, file)
        logger.info('13')
        Image.fromarray(tensor_to_bytes(images[0])).save(
            file_path,
            pnginfo=metadata,
            compress_level=4,
        )
        logger.info('14')
        output_files.append(file_path)
        logger.info('15')
        
        format_type, format_ext = format.split("/")
        logger.info('16')
        if format_type == "image":
            logger.info('17')
            file = f"{filename}_{counter:05}.{format_ext}"
            file_path = os.path.join(full_output_folder, file)
            images = tensor_to_bytes(images)
            if pingpong:
                images = np.concatenate((images, images[-2:0:-1]))
            frames = [Image.fromarray(f) for f in images]
            # Use pillow directly to save an animated image
            frames[0].save(
                file_path,
                format=format_ext.upper(),
                save_all=True,
                append_images=frames[1:],
                duration=round(1000 / frame_rate),
                loop=loop_count,
                compress_level=4,
            )
            output_files.append(file_path)
        else:
            logger.info('18ffmpeg')
            # Use ffmpeg to save a video
            if ffmpeg_path is None:
                #Should never be reachable
                raise ProcessLookupError("Could not find ffmpeg")
            logger.info('19')
            video_format_path = folder_paths.get_full_path("VHS_video_formats", format_ext + ".json")
            logger.info('20')
            with open(video_format_path, 'r') as stream:
                video_format = json.load(stream)
            logger.info('21')
            video_format = apply_format_widgets(format_ext, kwargs)
            logger.info('22')
            if video_format.get('input_color_depth', '8bit') == '16bit':
                images = tensor_to_shorts(images)
                i_pix_fmt = 'rgb48'
            else:
                images = tensor_to_bytes(images)
                i_pix_fmt = 'rgb24'
            logger.info('23')
            if pingpong:
                images = np.concatenate((images, images[-2:0:-1]))
            logger.info('24')
            file = f"{filename}_{counter:05}.{video_format['extension']}"
            logger.info('25')
            logger.info(file)
            file_path = os.path.join(full_output_folder, file)
            logger.info('26')
            dimensions = f"{len(images[0][0])}x{len(images[0])}"
            logger.info(dimensions)
            loop_args = ["-vf", "loop=loop=" + str(loop_count)+":size=" + str(len(images))]
            logger.info('27')
            logger.info(loop_args)
            bitrate_arg = []
            bitrate = video_format.get('bitrate')
            if bitrate is not None:
                bitrate_arg = ["-b:v", str(bitrate) + "M" if video_format.get('megabit') == 'True' else str(bitrate) + "K"]
            args = [ffmpeg_path, "-v", "error", "-f", "rawvideo", "-pix_fmt", i_pix_fmt,
                    "-s", dimensions, "-r", str(frame_rate), "-i", "-"] \
                    + loop_args + video_format['main_pass'] + bitrate_arg
            logger.info('28')
            env=os.environ.copy()
            if  "environment" in video_format:
                env.update(video_format["environment"])
            res = None
            logger.info('29')
            if video_format.get('save_metadata', 'False') != 'False':
                logger.info('30')
                os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
                metadata = json.dumps(video_metadata)
                metadata_path = os.path.join(folder_paths.get_temp_directory(), "metadata.txt")
                #metadata from file should  escape = ; # \ and newline
                metadata = metadata.replace("\\","\\\\")
                metadata = metadata.replace(";","\\;")
                metadata = metadata.replace("#","\\#")
                metadata = metadata.replace("=","\\=")
                metadata = metadata.replace("\n","\\\n")
                metadata = "comment=" + metadata
                with open(metadata_path, "w") as f:
                    f.write(";FFMETADATA1\n")
                    f.write(metadata)
                m_args = args[:1] + ["-i", metadata_path] + args[1:]
                try:
                    res = subprocess.run(m_args + [file_path], input=images.tobytes(),
                                         capture_output=True, check=True, env=env)
                except subprocess.CalledProcessError as e:
                    #Check if output file exists. If it does, the re-execution
                    #will also fail. This obscures the cause of the error
                    #and seems to never occur concurrent to the metadata issue
                    if os.path.exists(file_path):
                        raise Exception("An error occured in the ffmpeg subprocess:\n" \
                                + e.stderr.decode("utf-8"))
                    #Res was not set
                    print(e.stderr.decode("utf-8"), end="", file=sys.stderr)
                    logger.warn("An error occurred when saving with metadata")

            if not res:
                logger.info('31')
                logger.info(args + [file_path])
                try:
                    res = subprocess.run(args + [file_path], input=images.tobytes(),
                                         capture_output=True, check=True, env=env)
                except subprocess.CalledProcessError as e:
                    raise Exception("An error occured in the ffmpeg subprocess:\n" \
                            + e.stderr.decode("utf-8"))
            if res.stderr:
                print(res.stderr.decode("utf-8"), end="", file=sys.stderr)
            output_files.append(file_path)

            logger.info('32')

            # Audio Injection after video is created, saves additional video with -audio.mp4

            # Create audio file if input was provided
            if audio:
                logger.info('33')
                output_file_with_audio = f"{filename}_{counter:05}-audio.{video_format['extension']}"
                output_file_with_audio_path = os.path.join(full_output_folder, output_file_with_audio)
                if "audio_pass" not in video_format:
                    logger.warn("Selected video format does not have explicit audio support")
                    video_format["audio_pass"] = ["-c:a", "libopus"]


                # FFmpeg command with audio re-encoding
                #TODO: expose audio quality options if format widgets makes it in
                #Reconsider forcing apad/shortest
                mux_args = [ffmpeg_path, "-v", "error", "-n", "-i", file_path,
                            "-i", "-", "-c:v", "copy"] \
                            + video_format["audio_pass"] \
                            + ["-af", "apad", "-shortest", output_file_with_audio_path]

                logger.info(mux_args)
                logger.info('34')

                try:
                    res = subprocess.run(mux_args, input=audio(), env=env,
                                         capture_output=True, check=True)
                except subprocess.CalledProcessError as e:
                    raise Exception("An error occured in the ffmpeg subprocess:\n" \
                            + e.stderr.decode("utf-8"))
                if res.stderr:
                    print(res.stderr.decode("utf-8"), end="", file=sys.stderr)
                output_files.append(output_file_with_audio_path)
                #Return this file with audio to the webui.
                #It will be muted unless opened or saved with right click
                file = output_file_with_audio

        previews = [
            {
                "filename": file,
                "subfolder": subfolder,
                "type": "output" if save_output else "temp",
                "format": format,
            }
        ]
        return {"ui": {"gifs": previews}, "result": ((save_output, output_files),)}
    @classmethod
    def VALIDATE_INPUTS(self, format):
        return True

class LoadAudio:
    @classmethod
    def INPUT_TYPES(s):
        #Hide ffmpeg formats if ffmpeg isn't available
        return {
            "required": {
                "audio_file": ("STRING", {"default": "input/", "vhs_path_extensions": ['wav','mp3','ogg','m4a','flac']}),
                },
            "optional" : {"seek_seconds": ("FLOAT", {"default": 0, "min": 0})}
        }

    RETURN_TYPES = ("VHS_AUDIO",)
    RETURN_NAMES = ("audio",)
    CATEGORY = "SH Video Helper Suite 🎥🅥🅗🅢"
    FUNCTION = "load_audio"
    def load_audio(self, audio_file, seek_seconds):
        if audio_file is None or validate_path(audio_file) != True:
            raise Exception("audio_file is not a valid path: " + audio_file)
        #Eagerly fetch the audio since the user must be using it if the
        #node executes, unlike Load Video
        audio = get_audio(audio_file, start_time=seek_seconds)
        return (lambda : audio,)

    @classmethod
    def IS_CHANGED(s, audio_file, seek_seconds):
        return hash_path(audio_file)

    @classmethod
    def VALIDATE_INPUTS(s, audio_file, **kwargs):
        return validate_path(audio_file, allow_none=True)

NODE_CLASS_MAPPINGS = {
    "SH_VHS_VideoCombine": VideoCombine,
    "SH_VHS_LoadVideo": LoadVideoUpload,
    "SH_VHS_LoadVideoPath": LoadVideoPath,
    "SH_VHS_LoadImages": LoadImagesFromDirectoryUpload,
    "SH_VHS_LoadImagesPath": LoadImagesFromDirectoryPath,
    "SH_VHS_LoadAudio": LoadAudio,
    # Latent and Image nodes
    "SH_VHS_SplitLatents": SplitLatents,
    "SH_VHS_SplitImages": SplitImages,
    "SH_VHS_MergeLatents": MergeLatents,
    "SH_VHS_MergeImages": MergeImages,
    "SH_VHS_SelectEveryNthLatent": SelectEveryNthLatent,
    "SH_VHS_SelectEveryNthImage": SelectEveryNthImage,
    "SH_VHS_GetLatentCount": GetLatentCount,
    "SH_VHS_GetImageCount": GetImageCount,
    "SH_VHS_DuplicateLatents": DuplicateLatents,
    "SH_VHS_DuplicateImages": DuplicateImages,
    # Batched Nodes
    "SH_VHS_VAEEncodeBatched": VAEEncodeBatched,
    "SH_VHS_VAEDecodeBatched": VAEDecodeBatched,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SH_VHS_VideoCombine": "Video Combine 🎥🅥🅗🅢",
    "SH_VHS_LoadVideo": "Load Video (Upload) 🎥🅥🅗🅢",
    "SH_VHS_LoadVideoPath": "Load Video (Path) 🎥🅥🅗🅢",
    "SH_VHS_LoadImages": "Load Images (Upload) 🎥🅥🅗🅢",
    "SH_VHS_LoadImagesPath": "Load Images (Path) 🎥🅥🅗🅢",
    "SH_VHS_LoadAudio": "Load Audio (Path)🎥🅥🅗🅢",
    # Latent and Image nodes
    "SH_VHS_SplitLatents": "Split Latent Batch 🎥🅥🅗🅢",
    "SH_VHS_SplitImages": "Split Image Batch 🎥🅥🅗🅢",
    "SH_VHS_MergeLatents": "Merge Latent Batches 🎥🅥🅗🅢",
    "SH_VHS_MergeImages": "Merge Image Batches 🎥🅥🅗🅢",
    "SH_VHS_SelectEveryNthLatent": "Select Every Nth Latent 🎥🅥🅗🅢",
    "SH_VHS_SelectEveryNthImage": "Select Every Nth Image 🎥🅥🅗🅢",
    "SH_VHS_GetLatentCount": "Get Latent Count 🎥🅥🅗🅢",
    "SH_VHS_GetImageCount": "Get Image Count 🎥🅥🅗🅢",
    "SH_VHS_DuplicateLatents": "Duplicate Latent Batch 🎥🅥🅗🅢",
    "SH_VHS_DuplicateImages": "Duplicate Image Batch 🎥🅥🅗🅢",
    # Batched Nodes
    "SH_VHS_VAEEncodeBatched": "VAE Encode Batched 🎥🅥🅗🅢",
    "SH_VHS_VAEDecodeBatched": "VAE Decode Batched 🎥🅥🅗🅢",
}
