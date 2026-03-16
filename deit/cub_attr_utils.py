import os
import numpy as np
import pandas as pd


def load_topk_attr_ids(ranking_file, topk=100, sheet_name=None):
    """
    Đọc ranking file (.csv/.xlsx) và lấy top-k attr_id theo đúng thứ tự trong file.
    Yêu cầu có cột 'attr_id'.
    """
    if ranking_file is None or ranking_file == "":
        raise ValueError("ranking_file is empty")

    ext = os.path.splitext(ranking_file)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(ranking_file)
    elif ext in [".xlsx", ".xls"]:
        if sheet_name is not None and sheet_name != "":
            df = pd.read_excel(ranking_file, sheet_name=sheet_name)
        else:
            df = pd.read_excel(ranking_file)
    else:
        raise ValueError(f"Unsupported ranking file format: {ranking_file}")

    if "attr_id" not in df.columns:
        raise ValueError("ranking file must contain column 'attr_id'")

    attr_ids = df["attr_id"].tolist()[:topk]
    attr_ids = [int(x) for x in attr_ids]
    return attr_ids


def load_image_id_to_path(cub_root):
    """
    Đọc CUB_200_2011/images.txt
    return:
        image_id_to_relpath: dict[int] -> str
        relpath_to_image_id: dict[str] -> int
    """
    images_txt = os.path.join(cub_root, "images.txt")
    if not os.path.exists(images_txt):
        raise FileNotFoundError(images_txt)

    image_id_to_relpath = {}
    relpath_to_image_id = {}

    with open(images_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            image_id_str, rel_path = line.split(" ", 1)
            image_id = int(image_id_str)
            rel_path = rel_path.strip()

            image_id_to_relpath[image_id] = rel_path
            relpath_to_image_id[rel_path] = image_id

    return image_id_to_relpath, relpath_to_image_id


def load_cub_image_attr_map(cub_root, use_certainty=False, certainty_threshold=1):
    """
    Đọc file:
        CUB_200_2011/attributes/image_attribute_labels.txt

    Format mỗi dòng thường là:
        image_id attr_id is_present certainty_id time

    return:
        dict[image_id] = np.array shape [312]
    """
    attr_file = os.path.join(cub_root, "attributes", "image_attribute_labels.txt")
    if not os.path.exists(attr_file):
        raise FileNotFoundError(attr_file)

    imgid_to_attr = {}

    with open(attr_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue

            image_id = int(parts[0])
            attr_id = int(parts[1])          # 1..312
            is_present = int(parts[2])

            certainty_id = 4
            if len(parts) >= 4:
                certainty_id = int(parts[3])

            if image_id not in imgid_to_attr:
                imgid_to_attr[image_id] = np.zeros(312, dtype=np.float32)

            value = float(is_present)
            if use_certainty:
                if certainty_id >= certainty_threshold:
                    value = float(is_present)
                else:
                    value = 0.0

            imgid_to_attr[image_id][attr_id - 1] = value

    return imgid_to_attr


def select_attr_by_ids(full_attr_vec, attr_ids):
    """
    full_attr_vec: np.array [312]
    attr_ids: list 1-based attr_id
    return: np.array [len(attr_ids)]
    """
    if full_attr_vec is None:
        raise ValueError("full_attr_vec is None")
    if attr_ids is None:
        raise ValueError("attr_ids is None")

    indices = [int(aid) - 1 for aid in attr_ids]
    return full_attr_vec[indices].astype(np.float32)


def build_relpath_to_topk_attr(cub_root, attr_ids, use_certainty=False, certainty_threshold=1):
    """
    Map từ relative image path trong CUB -> top-k attr vector

    Ví dụ:
        '001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg'
        -> np.array [topk]
    """
    image_id_to_relpath, _ = load_image_id_to_path(cub_root)
    imgid_to_attr312 = load_cub_image_attr_map(
        cub_root,
        use_certainty=use_certainty,
        certainty_threshold=certainty_threshold
    )

    relpath_to_topk_attr = {}
    zero_vec = np.zeros(len(attr_ids), dtype=np.float32)

    for image_id, rel_path in image_id_to_relpath.items():
        full_vec = imgid_to_attr312.get(image_id, None)
        if full_vec is None:
            relpath_to_topk_attr[rel_path] = zero_vec.copy()
        else:
            relpath_to_topk_attr[rel_path] = select_attr_by_ids(full_vec, attr_ids)

    return relpath_to_topk_attr


def normalize_rel_path(path_str):
    """
    Chuẩn hoá path để so khớp Linux/Windows.
    """
    return path_str.replace("\\", "/").lstrip("./")