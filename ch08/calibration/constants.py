from imgaug.augmenters import imgcorruptlike as icl


CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
NUM_ENSEMBLE_MEMBERS = 3

NUM_BINS = 10

NUM_SUBSET = 1000

NUM_INFERENCES_BBB = 20

CORRUPTION_FUNCTIONS = [
    icl.GaussianNoise,
    icl.ShotNoise,
    icl.ImpulseNoise,
    icl.DefocusBlur,
    icl.GlassBlur,
    icl.MotionBlur,
    icl.ZoomBlur,
    icl.Snow,
    icl.Frost,
    icl.Fog,
    icl.Brightness,
    icl.Contrast,
    icl.ElasticTransform,
    icl.Pixelate,
    icl.JpegCompression,
]


NUM_TYPES = len(CORRUPTION_FUNCTIONS)
NUM_LEVELS = 5
