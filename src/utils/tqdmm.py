from tqdm import tqdm

TQDMM_KWARGS = {"ncols": 99, "leave": False}


def tqdmm(iterable, desc="", **kwargs):
    return tqdm(iterable, desc=desc, **(TQDMM_KWARGS | kwargs))
