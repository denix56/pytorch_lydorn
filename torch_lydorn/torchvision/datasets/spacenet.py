import fnmatch
import os.path
import pathlib
import sys
import time
import argparse
import math
import glob
import re
import random
from typing import Dict, Tuple, Union, Optional, Callable, Any, List
from functools import partial

import fiona
import rasterio as rio
import rasterio.transform
from fiona.transform import transform_geom
from fiona.errors import FionaValueError
from rasterio.crs import CRS
from rasterio.transform import Affine

import albumentations as A

import shapely.geometry
import multiprocess
import itertools
import skimage.io
import numpy as np

from tqdm import tqdm

import torch
import torch.utils.data
import torchvision
from torch import Tensor
from torch.utils.data import ConcatDataset, Subset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from lydorn_utils import run_utils, image_utils, polygon_utils, geo_utils
from lydorn_utils import print_utils
from lydorn_utils import python_utils

from torch_lydorn.torchvision.datasets import utils

from torchgeo.datasets import SpaceNet

IMG_SIZE = 512


def preprocess(x):
    # TODO: handle last channel
    if x.shape[0] == 4:
        x = x[:3]
    if len(x.shape) == 3:
        x = x.permute(1, 2, 0).numpy().astype(np.uint16)
    else:
        x = x.numpy().astype(np.uint8)
    return x


def clamp_image(image, *args, **kwargs):
    percentiles = np.nanpercentile(image, q=[2., 98.], axis=(0, 1))
    image = image.astype(np.float32)
    image = np.clip(image, percentiles[0], percentiles[1])
    if np.all(percentiles[0] != percentiles[1]):
        image = (image - percentiles[0]) / (percentiles[1] - percentiles[0] + 1e-5) * 255
    image = np.clip(image, 0, 255)
    image = np.nan_to_num(image, nan=0, posinf=255, neginf=0)
    return image.astype(np.uint8)


class SpaceNetV2(SpaceNet):
    def __init__(
        self,
        root: str,
        image: str,
        collections: List[str] = [],
        pre_transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        download: bool = False,
        api_key: Optional[str] = None,
        checksum: bool = False,
        mode: str = 'train'
    ) -> None:
        super().__init__(root, image, collections, transforms, download, api_key, checksum)
        self.pre_transforms = pre_transforms
        self.mode = mode

        if 'train' in mode:
            self.crop = A.RandomResizedCrop(IMG_SIZE, IMG_SIZE, scale=(0.08, 1.1))
        elif 'val' in mode:
            self.crop = A.CenterCrop(IMG_SIZE, IMG_SIZE)

        # self.cc = 0

    def _load_image(self, path: str) -> Tuple[Tensor, Affine, CRS]:
        """Load a single image.

        Args:
            path: path to the image

        Returns:
            the image
        """
        filename = os.path.join(path)
        with rio.open(filename) as img:
            array = img.read().astype(np.int32)
            tensor = torch.from_numpy(array).float()
            return tensor, img.transform, img.crs

    def _load_mask(
        self, path: str, tfm: Affine, raster_crs: CRS, shape: Tuple[int, int]
    ):
        """Rasterizes the dataset's labels (in geojson format).

        Args:
            path: path to the label
            tfm: transform of corresponding image
            shape: shape of corresponding image

        Returns:
            Tensor: label tensor
        """
        try:
            with fiona.open(path) as src:
                vector_crs = CRS(src.crs)
                if raster_crs == vector_crs:
                    labels = [shapely.geometry.shape(feature["geometry"]) for feature in src]
                else:
                    labels = [
                        shapely.geometry.shape(transform_geom(
                            vector_crs.to_string(),
                            raster_crs.to_string(),
                            feature["geometry"],
                        ))
                        for feature in src
                    ]
        except FionaValueError:
            labels = []

        has_multi = True
        while has_multi:
            labels_new = []
            has_multi = False
            for l in labels:
                if l.geom_type == 'MultiPolygon':
                    labels_new.extend(l.geoms)
                    has_multi = True
                elif l.geom_type == 'Polygon':
                    labels_new.append(l)
            labels = labels_new
        bbox = shapely.geometry.box(0, 0, shape[1]-1, shape[0]-1)
        for i, poly in enumerate(labels):
            xs, ys = list(zip(*poly.exterior.coords[:]))[:2]
            rows, cols = rasterio.transform.rowcol(tfm, xs, ys)
            exterior_coords = [(c, r) for r, c in zip(rows, cols)]
            interior_coords = []
            for interior in poly.interiors:
                xs, ys = list(zip(*interior.coords[:]))[:2]
                rows, cols = rasterio.transform.rowcol(tfm, xs, ys)
                interior_coords.append([(c, r) for r, c in zip(rows, cols)])
            try:
                poly = shapely.Polygon(exterior_coords, interior_coords)
                if poly.intersects(bbox):
                    labels[i] = poly.intersection(bbox)
                else:
                    labels[i] = shapely.Polygon()
            except:
                labels[i] = shapely.Polygon()
        labels = [l for l in labels if not l.is_empty and l.geom_type == 'Polygon' and l.area > 0]
        return labels

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, Union[Tensor, shapely.Polygon]]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        files = self.files[index]
        img, tfm, raster_crs = self._load_image(files["image_path"])
        ch, cw = self.chip_size[self.image]
        img = img[:ch, :cw]
        img = clamp_image(preprocess(img))

        tfm2 = Affine.identity()

        if np.min(img.shape[:2]) < IMG_SIZE:
            scale_factor = IMG_SIZE / np.min(img.shape[:2])
            img = A.smallest_max_size(img, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
            tfm2 = tfm2 * Affine.scale(1./scale_factor)
        #
        if 'train' in self.mode:
            params = self.crop.get_params_dependent_on_targets(dict(image=img))
            x1, y1, x2, y2 = A.get_random_crop_coords(*img.shape[:2], **params)
            img = A.random_crop(img, **params)
            h, w = img.shape[:2]
            img = A.resize(img, IMG_SIZE, IMG_SIZE, cv2.INTER_LINEAR)
            tfm2 = tfm2 * Affine.translation(x1, y1) * Affine.scale(w/IMG_SIZE, h/IMG_SIZE)
        else:
            x1, y1, x2, y2 = A.get_center_crop_coords(*img.shape[:2], IMG_SIZE, IMG_SIZE)
            img = A.center_crop(img, IMG_SIZE, IMG_SIZE)
            tfm2 = tfm2 * Affine.translation(x1, y1)

        tfm = tfm * tfm2

        h, w = img.shape[:2]
        polygons = self._load_mask(files["label_path"], tfm, raster_crs, (h, w))

        sample = dict(image=img,
                      image_filepath=files["image_path"],
                      gt_polygons=polygons,
                      name=pathlib.Path(files["image_path"]).parts[-2],
                      image_mean=np.array([0.485, 0.456, 0.406]),
                      image_std=np.array([0.229, 0.224, 0.225]))

        # os.makedirs('test_imgs', exist_ok=True)
        if self.pre_transforms is not None:
            # pre_path = os.path.join(str(pathlib.Path(files["label_path"]).parent), 'poly_data.npz')
            # if False:#os.path.exists(pre_path):
            #     sample |= np.load(pre_path)
            # else:
            sample = self.pre_transforms(sample)
            sample['class_freq'] = np.mean(sample["gt_polygons_image"], axis=(0, 1)) / 255
            sample['num'] = 1
                # keys = ['gt_polygons_image', 'distances', 'sizes', 'gt_crossfield_angle', 'class_freq', 'num']
                # save_data = {k: sample[k] for k in keys if k in sample}
                #np.savez(pre_path, **save_data)
            # cv2.imwrite(f'test_imgs/image_{self.cc}.png', cv2.cvtColor(sample['image'], cv2.COLOR_RGB2BGR))
            # cv2.imwrite(f'test_imgs/poly_image_{self.cc}.png', cv2.cvtColor(sample['gt_polygons_image'], cv2.COLOR_RGB2BGR))
            # alpha = 0.25
            # added_image = cv2.addWeighted(sample['gt_polygons_image'], alpha, sample['image'], 1 - alpha, 0)
            # cv2.imwrite(f"test_imgs/mix_{self.cc}.png", cv2.cvtColor(added_image, cv2.COLOR_RGB2BGR))

        if self.transforms is not None:
            sample = self.transforms(sample)
            # alpha = 0.25
            # poly = sample['gt_polygons_image'].permute(1, 2, 0).numpy()
            # img = sample['image'].permute(1, 2, 0).numpy()
            # added_image = cv2.addWeighted(poly, alpha, img, 1 - alpha, 0)
            # cv2.imwrite(f"test_imgs/mix_trans_{self.cc}.png", cv2.cvtColor(added_image, cv2.COLOR_RGB2BGR))

        # self.cc += 1
        # if self.cc > 20:
        #     raise NotImplementedError()

        return sample


class SpaceNet1(SpaceNetV2):
    """SpaceNet 1: Building Detection v1 Dataset.

    `SpaceNet 1 <https://spacenet.ai/spacenet-buildings-dataset-v1/>`_
    is a dataset of building footprints over the city of Rio de Janeiro.

    Dataset features:

    * No. of images: 6940 (8 Band) + 6940 (RGB)
    * No. of polygons: 382,534 building labels
    * Area Coverage: 2544 sq km
    * GSD: 1 m (8 band),  50 cm (rgb)
    * Chip size: 101 x 110 (8 band), 406 x 438 (rgb)

    Dataset format:

    * Imagery - Worldview-2 GeoTIFFs

        * 8Band.tif (Multispectral)
        * RGB.tif (Pansharpened RGB)

    * Labels - GeoJSON

        * labels.geojson

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/1807.01232

    """

    dataset_id = "spacenet1"
    imagery = {"rgb": "RGB.tif", "8band": "8Band.tif"}
    chip_size = {"rgb": (406, 438), "8band": (101, 110)}
    label_glob = "labels.geojson"
    collection_md5_dict = {"sn1_AOI_1_RIO": "e6ea35331636fa0c036c04b3d1cbf226"}

    def __init__(
        self,
        root: str = "data",
        image: str = "rgb",
        pre_transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        download: bool = False,
        api_key: Optional[str] = None,
        checksum: bool = False,
        *args,
        **kwargs
    ) -> None:
        """Initialize a new SpaceNet 1 Dataset instance.

        Args:
            root: root directory where dataset can be found
            image: image selection which must be "rgb" or "8band"
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version.
            download: if True, download dataset and store it in the root directory.
            api_key: a RadiantEarth MLHub API key to use for downloading the dataset
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing
        """
        collections = ["sn1_AOI_1_RIO"]
        assert image in {"rgb", "8band"}
        super().__init__(
            root, image, collections, pre_transforms, transforms, download, api_key, checksum, *args, **kwargs
        )


class SpaceNet2(SpaceNetV2):
    r"""SpaceNet 2: Building Detection v2 Dataset.

    `SpaceNet 2 <https://spacenet.ai/spacenet-buildings-dataset-v2/>`_
    is a dataset of building footprints over the cities of Las Vegas,
    Paris, Shanghai and Khartoum.

    Collection features:

    +------------+---------------------+------------+------------+
    |    AOI     | Area (km\ :sup:`2`\)| # Images   | # Buildings|
    +============+=====================+============+============+
    | Las Vegas  |    216              |   3850     |  151,367   |
    +------------+---------------------+------------+------------+
    | Paris      |    1030             |   1148     |  23,816    |
    +------------+---------------------+------------+------------+
    | Shanghai   |    1000             |   4582     |  92,015    |
    +------------+---------------------+------------+------------+
    | Khartoum   |    765              |   1012     |  35,503    |
    +------------+---------------------+------------+------------+

    Imagery features:

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1
        :stub-columns: 1

        *   -
            - PAN
            - MS
            - PS-MS
            - PS-RGB
        *   - GSD (m)
            - 0.31
            - 1.24
            - 0.30
            - 0.30
        *   - Chip size (px)
            - 650 x 650
            - 162 x 162
            - 650 x 650
            - 650 x 650

    Dataset format:

    * Imagery - Worldview-3 GeoTIFFs

        * PAN.tif (Panchromatic)
        * MS.tif (Multispectral)
        * PS-MS (Pansharpened Multispectral)
        * PS-RGB (Pansharpened RGB)

    * Labels - GeoJSON

        * label.geojson

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/1807.01232

    """

    dataset_id = "spacenet2"
    collection_md5_dict = {
        "sn2_AOI_2_Vegas": "a5a8de355290783b88ac4d69c7ef0694",
        "sn2_AOI_3_Paris": "8299186b7bbfb9a256d515bad1b7f146",
        "sn2_AOI_4_Shanghai": "4e3e80f2f437faca10ca2e6e6df0ef99",
        "sn2_AOI_5_Khartoum": "8070ff9050f94cd9f0efe9417205d7c3",
    }

    imagery = {
        "MS": "MS.tif",
        "PAN": "PAN.tif",
        "PS-MS": "PS-MS.tif",
        "PS-RGB": "PS-RGB.tif",
    }
    chip_size = {
        "MS": (162, 162),
        "PAN": (650, 650),
        "PS-MS": (650, 650),
        "PS-RGB": (650, 650),
    }
    label_glob = "label.geojson"

    def __init__(
        self,
        root: str = "data",
        image: str = "PS-RGB",
        collections: List[str] = [],
        pre_transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        download: bool = False,
        api_key: Optional[str] = None,
        checksum: bool = False,
        *args,
        **kwargs) -> None:
        """Initialize a new SpaceNet 2 Dataset instance.

        Args:
            root: root directory where dataset can be found
            image: image selection which must be in ["MS", "PAN", "PS-MS", "PS-RGB"]
            collections: collection selection which must be a subset of:
                         [sn2_AOI_2_Vegas, sn2_AOI_3_Paris, sn2_AOI_4_Shanghai,
                         sn2_AOI_5_Khartoum]. If unspecified, all collections will be
                         used.
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory.
            api_key: a RadiantEarth MLHub API key to use for downloading the dataset
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing
        """
        assert image in {"MS", "PAN", "PS-MS", "PS-RGB"}
        super().__init__(
            root, image, collections, pre_transforms, transforms, download, api_key, checksum, *args, **kwargs
        )


class SpaceNet4(SpaceNetV2):
    """SpaceNet 4: Off-Nadir Buildings Dataset.

    `SpaceNet 4 <https://spacenet.ai/off-nadir-building-detection/>`_ is a
    dataset of 27 WV-2 imagery captured at varying off-nadir angles and
    associated building footprints over the city of Atlanta. The off-nadir angle
    ranges from 7 degrees to 54 degrees.

    Dataset features:

    * No. of chipped images: 28,728 (PAN/MS/PS-RGBNIR)
    * No. of label files: 1064
    * No. of building footprints: >120,000
    * Area Coverage: 665 sq km
    * Chip size: 225 x 225 (MS), 900 x 900 (PAN/PS-RGBNIR)

    Dataset format:

    * Imagery - Worldview-2 GeoTIFFs

        * PAN.tif (Panchromatic)
        * MS.tif (Multispectral)
        * PS-RGBNIR (Pansharpened RGBNIR)

    * Labels - GeoJSON

        * labels.geojson

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/1903.12239

    """

    dataset_id = "spacenet4"
    collection_md5_dict = {"sn4_AOI_6_Atlanta": "c597d639cba5257927a97e3eff07b753"}

    imagery = {"MS": "MS.tif", "PAN": "PAN.tif", "PS-RGBNIR": "PS-RGBNIR.tif"}
    chip_size = {"MS": (225, 225), "PAN": (900, 900), "PS-RGBNIR": (900, 900)}
    label_glob = "labels.geojson"

    angle_catalog_map = {
        "nadir": [
            "1030010003D22F00",
            "10300100023BC100",
            "1030010003993E00",
            "1030010003CAF100",
            "1030010002B7D800",
            "10300100039AB000",
            "1030010002649200",
            "1030010003C92000",
            "1030010003127500",
            "103001000352C200",
            "103001000307D800",
        ],
        "off-nadir": [
            "1030010003472200",
            "1030010003315300",
            "10300100036D5200",
            "103001000392F600",
            "1030010003697400",
            "1030010003895500",
            "1030010003832800",
        ],
        "very-off-nadir": [
            "10300100035D1B00",
            "1030010003CCD700",
            "1030010003713C00",
            "10300100033C5200",
            "1030010003492700",
            "10300100039E6200",
            "1030010003BDDC00",
            "1030010003CD4300",
            "1030010003193D00",
        ],
    }

    def __init__(
        self,
        root: str = "data",
        image: str = "PS-RGBNIR",
        angles: List[str] = [],
        pre_transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        download: bool = False,
        api_key: Optional[str] = None,
        checksum: bool = False,
        *args,
        **kwargs
    ) -> None:
        """Initialize a new SpaceNet 4 Dataset instance.

        Args:
            root: root directory where dataset can be found
            image: image selection which must be in ["MS", "PAN", "PS-RGBNIR"]
            angles: angle selection which must be in ["nadir", "off-nadir",
                "very-off-nadir"]
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory.
            api_key: a RadiantEarth MLHub API key to use for downloading the dataset
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing
        """
        collections = ["sn4_AOI_6_Atlanta"]
        assert image in {"MS", "PAN", "PS-RGBNIR"}
        self.angles = angles
        if self.angles:
            for angle in self.angles:
                assert angle in self.angle_catalog_map.keys()
        super().__init__(
            root, image, collections, pre_transforms, transforms, download, api_key, checksum, *args, **kwargs
        )

    def _load_files(self, root: str) -> List[Dict[str, str]]:
        """Return the paths of the files in the dataset.

        Args:
            root: root dir of dataset

        Returns:
            list of dicts containing paths for each pair of image and label
        """
        files = []
        nadir = []
        offnadir = []
        veryoffnadir = []
        images = glob.glob(os.path.join(root, self.collections[0], "*", self.filename))
        images = sorted(images)

        catalog_id_pattern = re.compile(r"(_[A-Z0-9])\w+$")
        for imgpath in images:
            imgdir = os.path.basename(os.path.dirname(imgpath))
            match = catalog_id_pattern.search(imgdir)
            assert match is not None, "Invalid image directory"
            catalog_id = match.group()[1:]

            lbl_dir = os.path.dirname(imgpath).split("-nadir")[0]

            lbl_path = os.path.join(f"{lbl_dir}-labels", self.label_glob)
            assert os.path.exists(lbl_path)

            _file = {"image_path": imgpath, "label_path": lbl_path}
            if catalog_id in self.angle_catalog_map["very-off-nadir"]:
                veryoffnadir.append(_file)
            elif catalog_id in self.angle_catalog_map["off-nadir"]:
                offnadir.append(_file)
            elif catalog_id in self.angle_catalog_map["nadir"]:
                nadir.append(_file)

        angle_file_map = {
            "nadir": nadir,
            "off-nadir": offnadir,
            "very-off-nadir": veryoffnadir,
        }

        if not self.angles:
            files.extend(nadir + offnadir + veryoffnadir)
        else:
            for angle in self.angles:
                files.extend(angle_file_map[angle])
        return files



# def get_transforms(mode: str, test_img_size: int = None, resize_to_min_size: bool = False, target_domain_root: str = None,
#                    transforms_dict=None):
#     if transforms_dict is None:
#         transforms_dict = dict(histogram_matching=0.5,
#                               downscale=0.5,
#                               image_compression=0.5,
#                               clahe=0.0,
#                               color_jitter=0.5,
#                               blur=0.5,
#                               gauss_noise=0.5,
#                               iso_noise=0.5,
#                               hflip=0.5,
#                               vflip=0.5,
#                               gray=0.0)
#     if 'train' in mode:
#         if target_domain_root is not None:
#             img_list = [os.path.join(target_domain_root, p) for p in sorted(os.listdir(target_domain_root))]
#             domain_augs = A.HistogramMatching(img_list, p=transforms_dict['histogram_matching'])
#         else:
#             domain_augs = A.NoOp()
#
#         perm_transforms = A.OneOf([A.Compose(t) for t in [[
#             A.Downscale(scale_min=0.05, scale_max=0.5, interpolation=cv2.INTER_CUBIC, p=transforms_dict['downscale']),
#             A.ImageCompression(quality_lower=50, p=transforms_dict['image_compression']),
#             A.CLAHE(p=transforms_dict['clahe']),
#             A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.01, hue=0.01, p=transforms_dict['color_jitter']),
#             A.OneOf([
#                 A.GaussianBlur(),
#                 A.MedianBlur()], p=transforms_dict['blur']),
#             A.GaussNoise(p=transforms_dict['gauss_noise']),
#             A.ISONoise(p=transforms_dict['iso_noise'])
#         ]]], p=1.0)
#
#         transforms = A.Compose([
#             A.Lambda(image=clamp_image),
#             A.RandomResizedCrop(512, 512, ratio=(0.85, 1.15)),
#             domain_augs,
#             perm_transforms,
#             A.ToGray(p=transforms_dict['gray']),
#             A.HorizontalFlip(p=transforms_dict['hflip']),
#             A.VerticalFlip(p=transforms_dict['vflip']),
#             A.Affine(translate_percent=(-0.3, 0.3), rotate=(-90, 90), shear=(5, 5), scale=(0.2, 1.1), cval=0,
#                      cval_mask=-1, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST, keep_ratio=True),
#             A.Normalize(),
#             ToTensorV2()
#         ], additional_targets={'cls_mask': 'mask', 'osm_mask': 'mask'})
#     else:
#         transforms = A.Compose([
#             A.Lambda(image=clamp_image),
#             A.SmallestMaxSize(512) if resize_to_min_size else A.NoOp(),
#             A.CenterCrop(test_img_size, test_img_size),
#             A.Normalize(),
#             ToTensorV2()
#         ], additional_targets={'cls_mask': 'mask', 'osm_mask': 'mask'})
#
#     trans_wrapper = transforms
#     return trans_wrapper


def get_spacenet_splits(data_root: str, api_key: str, train_part: float, version: int):
    split_path = os.path.join(data_root, f'spacenet{version}_split.npz')
    if os.path.exists(split_path):
        splits = np.load(split_path)
    else:
        if version == 1:
            dataset = SpaceNet1(root=data_root, transforms=None, download=True, api_key=api_key)
        elif version == 2:
            dataset = SpaceNet2(root=data_root, transforms=None, download=True, api_key=api_key)
        elif version == 4:
            dataset = SpaceNet4(root=data_root, angles=['nadir'], download=True, api_key=api_key)
        else:
            raise NotImplementedError()
        indices = np.arange(len(dataset))
        np.random.default_rng().shuffle(indices)
        arr_len = indices.shape[0]
        splits = {'train': indices[:int(train_part*arr_len)],
                  'val': indices[int(train_part*arr_len):]}
        np.savez(split_path, **splits)
    return splits


def get_spacenet(data_root: str, api_key: str, mode: str, train_part: float = 0.95,
                 target_domain_root: str = None, pre_transform=None, transform=None):
    datasets = []
    # Issue with downloading v4
    ds_class_dict = {
        1: SpaceNet1,
        2: SpaceNet2,
        4: partial(SpaceNet4, angles=['nadir']),
    }

    for version in [1, 2, 4]:
        if version == 4 and 'train' not in mode:
            continue

        ds = ds_class_dict[version](root=data_root,
                                    pre_transforms=pre_transform,
                                    transforms=transform,
                                    download=True, api_key=api_key, mode=mode)
        if mode != 'train_val':
            splits = get_spacenet_splits(data_root, api_key, train_part, version=version)
            ds = Subset(ds, splits[mode])
        datasets.append(ds)
    dataset = ConcatDataset(datasets)
    return dataset


# def main():
#     # Test using transforms from the frame_field_learning project:
#     from frame_field_learning import data_transforms
#
#     config = {
#         "data_dir_candidates": [
#             "/data/titane/user/nigirard/data",
#             "~/data",
#             "/data"
#         ],
#         "dataset_params": {
#             "root_dirname": "AerialImageDataset",
#             "pre_process": False,
#             "gt_source": "disk",
#             "gt_type": "tif",
#             "gt_dirname": "gt",
#             "mask_only": False,
#             "small": True,
#             "data_patch_size": 425,
#             "input_patch_size": 300,
#
#             "train_fraction": 0.75
#         },
#         "num_workers": 8,
#         "data_aug_params": {
#             "enable": True,
#             "vflip": True,
#             "affine": True,
#             "scaling": [0.9, 1.1],
#             "color_jitter": True,
#             "device": "cuda"
#         }
#     }
#
#     # Find data_dir
#     data_dir = python_utils.choose_first_existing_path(config["data_dir_candidates"])
#     if data_dir is None:
#         print_utils.print_error("ERROR: Data directory not found!")
#         exit()
#     else:
#         print_utils.print_info("Using data from {}".format(data_dir))
#     root_dir = os.path.join(data_dir, config["dataset_params"]["root_dirname"])
#
#     # --- Transforms: --- #
#     # --- pre-processing transform (done once then saved on disk):
#     # --- Online transform done on the host (CPU):
#     online_cpu_transform = data_transforms.get_online_cpu_transform(config,
#                                                                     augmentations=config["data_aug_params"]["enable"])
#     train_online_cuda_transform = data_transforms.get_online_cuda_transform(config, augmentations=config["data_aug_params"]["enable"])
#     mask_only = config["dataset_params"]["mask_only"]
#     kwargs = {
#         "pre_process": config["dataset_params"]["pre_process"],
#         "transform": online_cpu_transform,
#         "patch_size": config["dataset_params"]["data_patch_size"],
#         "patch_stride": config["dataset_params"]["input_patch_size"],
#         "pre_transform": data_transforms.get_offline_transform_patch(distances=not mask_only, sizes=not mask_only),
#         "small": config["dataset_params"]["small"],
#         "pool_size": config["num_workers"],
#         "gt_source": config["dataset_params"]["gt_source"],
#         "gt_type": config["dataset_params"]["gt_type"],
#         "gt_dirname": config["dataset_params"]["gt_dirname"],
#         "mask_only": config["dataset_params"]["mask_only"],
#     }
#     train_val_split_point = config["dataset_params"]["train_fraction"] * 36
#     def train_tile_filter(tile): return tile["number"] <= train_val_split_point
#     def val_tile_filter(tile): return train_val_split_point < tile["number"]
#     # --- --- #
#     fold = "train"
#     if fold == "train":
#         dataset = InriaAerial(root_dir, fold="train", tile_filter=train_tile_filter, **kwargs)
#     elif fold == "val":
#         dataset = InriaAerial(root_dir, fold="train", tile_filter=val_tile_filter, **kwargs)
#     elif fold == "test":
#         dataset = InriaAerial(root_dir, fold="test", **kwargs)
#
#     print(f"dataset has {len(dataset)} samples.")
#     print("# --- Sample 0 --- #")
#     sample = dataset[0]
#     for key, item in sample.items():
#         print("{}: {}".format(key, type(item)))
#
#     print("# --- Samples --- #")
#     # for data in tqdm(dataset):
#     #     pass
#
#     data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config["num_workers"])
#     print("# --- Batches --- #")
#     for batch in tqdm(data_loader):
#
#         # batch["distances"] = batch["distances"].float()
#         # batch["sizes"] = batch["sizes"].float()
#
#         # im = np.array(batch["image"][0])
#         # im = np.moveaxis(im, 0, -1)
#         # skimage.io.imsave('im_before_transform.png', im)
#         #
#         # distances = np.array(batch["distances"][0])
#         # distances = np.moveaxis(distances, 0, -1)
#         # skimage.io.imsave('distances_before_transform.png', distances)
#         #
#         # sizes = np.array(batch["sizes"][0])
#         # sizes = np.moveaxis(sizes, 0, -1)
#         # skimage.io.imsave('sizes_before_transform.png', sizes)
#
#         print("----")
#         print(batch["name"])
#
#         print("image:", batch["image"].shape, batch["image"].min().item(), batch["image"].max().item())
#         im = np.array(batch["image"][0])
#         im = np.moveaxis(im, 0, -1)
#         skimage.io.imsave('im.png', im)
#
#         if "gt_polygons_image" in batch:
#             print("gt_polygons_image:", batch["gt_polygons_image"].shape, batch["gt_polygons_image"].min().item(),
#                   batch["gt_polygons_image"].max().item())
#             seg = np.array(batch["gt_polygons_image"][0]) / 255
#             seg = np.moveaxis(seg, 0, -1)
#             seg_display = utils.get_seg_display(seg)
#             seg_display = (seg_display * 255).astype(np.uint8)
#             skimage.io.imsave("gt_seg.png", seg_display)
#
#         if "gt_crossfield_angle" in batch:
#             print("gt_crossfield_angle:", batch["gt_crossfield_angle"].shape, batch["gt_crossfield_angle"].min().item(),
#                   batch["gt_crossfield_angle"].max().item())
#             gt_crossfield_angle = np.array(batch["gt_crossfield_angle"][0])
#             gt_crossfield_angle = np.moveaxis(gt_crossfield_angle, 0, -1)
#             skimage.io.imsave('gt_crossfield_angle.png', gt_crossfield_angle)
#
#         if "distances" in batch:
#             print("distances:", batch["distances"].shape, batch["distances"].min().item(), batch["distances"].max().item())
#             distances = np.array(batch["distances"][0])
#             distances = np.moveaxis(distances, 0, -1)
#             skimage.io.imsave('distances.png', distances)
#
#         if "sizes" in batch:
#             print("sizes:", batch["sizes"].shape, batch["sizes"].min().item(), batch["sizes"].max().item())
#             sizes = np.array(batch["sizes"][0])
#             sizes = np.moveaxis(sizes, 0, -1)
#             skimage.io.imsave('sizes.png', sizes)
#
#         # valid_mask = np.array(batch["valid_mask"][0])
#         # valid_mask = np.moveaxis(valid_mask, 0, -1)
#         # skimage.io.imsave('valid_mask.png', valid_mask)
#
#         print("Apply online tranform:")
#         batch = utils.batch_to_cuda(batch)
#         batch = train_online_cuda_transform(batch)
#         batch = utils.batch_to_cpu(batch)
#
#         print("image:", batch["image"].shape, batch["image"].min().item(), batch["image"].max().item())
#         print("gt_polygons_image:", batch["gt_polygons_image"].shape, batch["gt_polygons_image"].min().item(), batch["gt_polygons_image"].max().item())
#         print("gt_crossfield_angle:", batch["gt_crossfield_angle"].shape, batch["gt_crossfield_angle"].min().item(), batch["gt_crossfield_angle"].max().item())
#         # print("distances:", batch["distances"].shape, batch["distances"].min().item(), batch["distances"].max().item())
#         # print("sizes:", batch["sizes"].shape, batch["sizes"].min().item(), batch["sizes"].max().item())
#
#         # Save output to visualize
#         seg = np.array(batch["gt_polygons_image"][0])
#         seg = np.moveaxis(seg, 0, -1)
#         seg_display = utils.get_seg_display(seg)
#         seg_display = (seg_display * 255).astype(np.uint8)
#         skimage.io.imsave("gt_seg.png", seg_display)
#
#         im = np.array(batch["image"][0])
#         im = np.moveaxis(im, 0, -1)
#         skimage.io.imsave('im.png', im)
#
#         gt_crossfield_angle = np.array(batch["gt_crossfield_angle"][0])
#         gt_crossfield_angle = np.moveaxis(gt_crossfield_angle, 0, -1)
#         skimage.io.imsave('gt_crossfield_angle.png', gt_crossfield_angle)
#
#         distances = np.array(batch["distances"][0])
#         distances = np.moveaxis(distances, 0, -1)
#         skimage.io.imsave('distances.png', distances)
#
#         sizes = np.array(batch["sizes"][0])
#         sizes = np.moveaxis(sizes, 0, -1)
#         skimage.io.imsave('sizes.png', sizes)
#
#         # valid_mask = np.array(batch["valid_mask"][0])
#         # valid_mask = np.moveaxis(valid_mask, 0, -1)
#         # skimage.io.imsave('valid_mask.png', valid_mask)
#
#         input("Press enter to continue...")

