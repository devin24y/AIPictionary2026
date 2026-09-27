from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import torch
from PIL import Image

import config


LOGGER = logging.getLogger(__name__)


QUICKDRAW_LABELS: tuple[str, ...] = (
    "aircraft carrier",
    "airplane",
    "alarm clock",
    "ambulance",
    "angel",
    "animal migration",
    "ant",
    "anvil",
    "apple",
    "arm",
    "asparagus",
    "axe",
    "backpack",
    "banana",
    "bandage",
    "barn",
    "baseball",
    "baseball bat",
    "basket",
    "basketball",
    "bat",
    "bathtub",
    "beach",
    "bear",
    "beard",
    "bed",
    "bee",
    "belt",
    "bench",
    "bicycle",
    "binoculars",
    "bird",
    "birthday cake",
    "blackberry",
    "blueberry",
    "book",
    "boomerang",
    "bottlecap",
    "bowtie",
    "bracelet",
    "brain",
    "bread",
    "bridge",
    "broccoli",
    "broom",
    "bucket",
    "bulldozer",
    "bus",
    "bush",
    "butterfly",
    "cactus",
    "cake",
    "calculator",
    "calendar",
    "camel",
    "camera",
    "camouflage",
    "campfire",
    "candle",
    "cannon",
    "canoe",
    "car",
    "carrot",
    "castle",
    "cat",
    "ceiling fan",
    "cell phone",
    "cello",
    "chair",
    "chandelier",
    "church",
    "circle",
    "clarinet",
    "clock",
    "cloud",
    "coffee cup",
    "compass",
    "computer",
    "cookie",
    "cooler",
    "couch",
    "cow",
    "crab",
    "crayon",
    "crocodile",
    "crown",
    "cruise ship",
    "cup",
    "diamond",
    "dishwasher",
    "diving board",
    "dog",
    "dolphin",
    "donut",
    "door",
    "dragon",
    "dresser",
    "drill",
    "drums",
    "duck",
    "dumbbell",
    "ear",
    "elbow",
    "elephant",
    "envelope",
    "eraser",
    "eye",
    "eyeglasses",
    "face",
    "fan",
    "feather",
    "fence",
    "finger",
    "fire hydrant",
    "fireplace",
    "firetruck",
    "fish",
    "flamingo",
    "flashlight",
    "flip flops",
    "floor lamp",
    "flower",
    "flying saucer",
    "foot",
    "fork",
    "frog",
    "frying pan",
    "garden",
    "garden hose",
    "giraffe",
    "goatee",
    "golf club",
    "grapes",
    "grass",
    "guitar",
    "hamburger",
    "hammer",
    "hand",
    "harp",
    "hat",
    "headphones",
    "hedgehog",
    "helicopter",
    "helmet",
    "hexagon",
    "hockey puck",
    "hockey stick",
    "horse",
    "hospital",
    "hot air balloon",
    "hot dog",
    "hot tub",
    "hourglass",
    "house",
    "house plant",
    "hurricane",
    "ice cream",
    "jacket",
    "jail",
    "kangaroo",
    "key",
    "keyboard",
    "knee",
    "knife",
    "ladder",
    "lantern",
    "laptop",
    "leaf",
    "leg",
    "light bulb",
    "lighthouse",
    "lightning",
    "line",
    "lion",
    "lipstick",
    "lobster",
    "lollipop",
    "mailbox",
    "map",
    "marker",
    "matches",
    "megaphone",
    "mermaid",
    "microphone",
    "microwave",
    "monkey",
    "moon",
    "mosquito",
    "motorbike",
    "mountain",
    "mouse",
    "moustache",
    "mouth",
    "mug",
    "mushroom",
    "nail",
    "necklace",
    "nose",
    "ocean",
    "octagon",
    "octopus",
    "onion",
    "oven",
    "owl",
    "paint can",
    "paintbrush",
    "palm tree",
    "panda",
    "pants",
    "paper clip",
    "parachute",
    "parrot",
    "passport",
    "peanut",
    "pear",
    "peas",
    "pencil",
    "penguin",
    "piano",
    "pickup truck",
    "picture frame",
    "pig",
    "pillow",
    "pineapple",
    "pizza",
    "pliers",
    "police car",
    "pond",
    "pool",
    "popsicle",
    "postcard",
    "potato",
    "power outlet",
    "purse",
    "rabbit",
    "raccoon",
    "radio",
    "rain",
    "rainbow",
    "rake",
    "remote control",
    "rhinoceros",
    "rifle",
    "river",
    "roller coaster",
    "rollerskates",
    "sailboat",
    "sandwich",
    "saw",
    "saxophone",
    "school bus",
    "scissors",
    "scorpion",
    "screwdriver",
    "sea turtle",
    "see saw",
    "shark",
    "sheep",
    "shoe",
    "shorts",
    "shovel",
    "sink",
    "skateboard",
    "skull",
    "skyscraper",
    "sleeping bag",
    "smiley face",
    "sn",
    "snake",
    "snorkel",
    "snowflake",
    "snowman",
    "soccer ball",
    "sock",
    "speedboat",
    "spider",
    "spoon",
    "spreadsheet",
    "square",
    "squiggle",
    "squirrel",
    "stairs",
    "star",
    "steak",
    "stereo",
    "stethoscope",
    "stitches",
    "stop sign",
    "stove",
    "strawberry",
    "streetlight",
    "string bean",
    "submarine",
    "suitcase",
    "sun",
    "swan",
    "sweater",
    "swing set",
    "sword",
    "t-shirt",
    "table",
    "teapot",
    "teddy-bear",
    "telephone",
    "television",
    "tennis racquet",
    "tent",
    "The Eiffel Tower",
    "The Great Wall of China",
    "The Mona Lisa",
    "tiger",
    "toaster",
    "toe",
    "toilet",
    "toothbrush",
    "toothpaste",
    "tornado",
    "tractor",
    "traffic light",
    "train",
    "tree",
    "triangle",
    "trombone",
    "truck",
    "trumpet",
    "umbrella",
    "underwear",
    "van",
    "vase",
    "violin",
    "washing machine",
    "watermelon",
    "waterslide",
    "whale",
    "wheel",
    "windmill",
    "wine bottle",
    "wine glass",
    "wristwatch",
    "yoga",
    "zebra",
    "zigzag",
)


class DrawingGuesser:
    """
    Zero-shot QuickDraw classifier backed by OpenCLIP.

    Model downloading and initialization should occur inside the background
    inference thread. If OpenCLIP cannot initialize, the class falls back to a
    pretrained MobileNetV3 ImageNet classifier instead of terminating the app.
    """

    def __init__(self) -> None:
        torch.set_num_threads(config.TORCH_CPU_THREADS)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.backend = ""
        self.model = None
        self.preprocess = None
        self.text_features: torch.Tensor | None = None
        self.labels: Sequence[str] = ()
        self.initialization_warning = ""

        try:
            self._initialize_clip()
        except Exception as clip_error:
            LOGGER.exception("OpenCLIP initialization failed")
            self.initialization_warning = (
                f"OpenCLIP unavailable: {type(clip_error).__name__}. "
                "Using ImageNet fallback."
            )
            self._initialize_mobilenet_fallback()

    def _initialize_clip(self) -> None:
        import open_clip

        LOGGER.info(
            "Loading OpenCLIP %s with %s weights on %s",
            config.CLIP_MODEL_NAME,
            config.CLIP_PRETRAINED_WEIGHTS,
            self.device,
        )

        model, _, preprocess = open_clip.create_model_and_transforms(
            config.CLIP_MODEL_NAME,
            pretrained=config.CLIP_PRETRAINED_WEIGHTS,
            cache_dir=str(config.MODEL_CACHE_DIR),
            device=self.device,
        )
        tokenizer = open_clip.get_tokenizer(config.CLIP_MODEL_NAME)

        model.eval()
        self.model = model
        self.preprocess = preprocess
        self.labels = QUICKDRAW_LABELS

        prompt_features = []

        with torch.inference_mode():
            for template in config.CLIP_PROMPT_TEMPLATES:
                prompts = [template.format(label) for label in self.labels]
                tokenized = tokenizer(prompts).to(self.device)
                features = model.encode_text(tokenized)
                features = features / features.norm(dim=-1, keepdim=True)
                prompt_features.append(features)

            averaged_features = torch.stack(
                prompt_features,
                dim=0,
            ).mean(dim=0)
            averaged_features = averaged_features / averaged_features.norm(
                dim=-1,
                keepdim=True,
            )

        self.text_features = averaged_features
        self.backend = "OpenCLIP QuickDraw zero-shot"

    def _initialize_mobilenet_fallback(self) -> None:
        from torchvision.models import (
            MobileNet_V3_Small_Weights,
            mobilenet_v3_small,
        )

        weights = MobileNet_V3_Small_Weights.DEFAULT
        model = mobilenet_v3_small(weights=weights)
        model.eval()
        model.to(self.device)

        self.model = model
        self.preprocess = weights.transforms()
        self.labels = tuple(weights.meta["categories"])
        self.text_features = None
        self.backend = "MobileNetV3 ImageNet fallback"

    def predict(
        self,
        image: np.ndarray,
    ) -> list[tuple[str, float]]:
        """
        Returns the top predictions as (label, percentage) tuples.
        """

        if image is None or image.size == 0:
            return []

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected an RGB image with shape HxWx3")

        pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
        input_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            if self.text_features is not None:
                image_features = self.model.encode_image(input_tensor)
                image_features = image_features / image_features.norm(
                    dim=-1,
                    keepdim=True,
                )

                logits = 100.0 * image_features @ self.text_features.T
            else:
                logits = self.model(input_tensor)

            probabilities = logits.softmax(dim=-1)[0]
            top_count = min(
                config.TOP_K_PREDICTIONS,
                int(probabilities.shape[0]),
            )
            values, indices = probabilities.topk(top_count)

        results: list[tuple[str, float]] = []
        for score, index in zip(
            values.detach().cpu().tolist(),
            indices.detach().cpu().tolist(),
        ):
            results.append(
                (
                    self.labels[int(index)],
                    float(score) * 100.0,
                )
            )

        return results