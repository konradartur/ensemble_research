import os

RANDOM_SEED = 420

DATA_DIR = os.path.join(os.environ['HOME'], 'data')

RESULTS_DIR = os.path.join(os.environ['HOME'], 'results')

DISRUPTIONS = [
        "brightness",
        "contrast",
        "defocus_blur",
        "elastic_transform",
        "fog",
        "frost",
        "gaussian_blur",
        "gaussian_noise",
        "glass_blur",
        "impulse_noise",
        "jpeg_compression",
        "motion_blur",
        "pixelate",
        "saturate",
        "shot_noise",
        "snow",
        "spatter",
        "speckle_noise",
        "zoom_blur"
]
