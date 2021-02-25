#!/usr/bin/env python3

import rasterio as rio
import numpy as np
from rasterio.warp import reproject, Resampling
from skimage.exposure import rescale_intensity

class LANDSAT8_PRE(object):
    def __init__(self, landsat8_id):

        pathrow = landsat8_id.split('_')[2]
        self.sceneid = landsat8_id
        self.path = pathrow[:3]
        self.row = pathrow[3:]
        self.rgb = [2,1,0,3]
        bands = []
        multi_prof = None
        for band in range(2,6):
            band_image, band_profile = self._get_band(band)
            bands.append(band_image)
            if not multi_prof:
                multi_prof = band_profile

        self.multi_prof = multi_prof
        mul_shape = bands[-1].shape
        self.mul_bands = np.zeros((len(bands), mul_shape[0], mul_shape[1]), dtype = bands[-1].dtype)
        for i in range(len(bands)):
            self.mul_bands[i] = bands[i]

        del bands

        self.pan_band, self.pan_prof = self._get_band(8)
        self.qa_band, self.qa_prof = self._get_band('QA')
        self.snow_ice_percent, self.cloud_mask, self.snow_mask, self.fill_mask = self._calc_cloud_ice_percent(self.qa_band)

        return

    def _get_band(self, bandname):

        url_base = 'https://s3-us-west-2.amazonaws.com/landsat-pds/c1/L8/{}/{}/{}/{}'
        filename = '{}_B{}.TIF'.format(self.sceneid, bandname)
        download_url = url_base.format(self.path,self.row,self.sceneid,filename)
        with rio.open(download_url) as src:
            band = src.read(1)
            band_profile = src.profile

        return band, band_profile

    def _calc_cloud_ice_percent(self, qa):

        cloud_high_conf = int('0000000001100000',2)
        snow_high_conf = int('0000011000000000',2)
        fill_pixels = int('0000000000000001', 2)
        cloud_mask = np.bitwise_and(qa, cloud_high_conf) == cloud_high_conf
        snow_mask = np.bitwise_and(qa, snow_high_conf) == snow_high_conf
        fill_mask = np.bitwise_and(qa, fill_pixels) == fill_pixels

        return np.true_divide(np.sum(cloud_mask | snow_mask),qa.size - np.sum(fill_mask)) * 100.0, cloud_mask, snow_mask, fill_mask

    @staticmethod
    def _get_threshold(band):

        return np.percentile(tmp[np.logical_and(tmp>0, tmp<65535)], (0,99))

    @staticmethod
    def _rescale_band(band, mask):
        tmp = band[np.logical_not(mask)]
        p_low, p_high = LANDSAT8_PRE._get_threshold(tmp)
        tmp = np.zeros(band.shape,dtype=np.uint16)
        tmp[np.logical_not(mask)] = rescale_intensity(band[np.logical_not(mask)],in_range=(p_low,p_high), out_range=(p_low, 63000))
        tmp[mask] = rescale_intensity(band[mask], in_range=(p_high,65535), out_range=(63000,65535))
        return tmp


class ASNARO1_L1B_PRE(object):
    def __init__(self, multi, pan):

        with rio.open(multi) as multi_src:
            self.multi_bands = multi_src.read()
            self.multi_prof = multi_src.profile
        with rio.open(pan) as pan_src:
            self.pan_band = pan_src.read(1)
            self.pan_prof = pan_src.profile

        self.rgb=[3,2,1,4]
        return

class Pansharpen(object):
    def __init__(self, multi, multi_prof, pan, pan_prof, rgb=[3,2,1,4], bweight=0.2, irweight=0.1):

        self.rgb = rgb
        self.multi_bands = multi
        self.multi_prof = multi_prof
        self.pan_band = pan
        self.pan_prof = pan_prof

        self.multi_bands = Pansharpen._upper_scale(self.multi_bands, self.pan_band, self.multi_prof, self.pan_prof)
        self.bweight = bweight
        self.irweight = irweight

        return

    @staticmethod
    def _grey_world(bands):
        mu_g = np.average(bands[1])
        bands[0] = np.minimum(bands[0] * (mu_g / np.average(bands[0])), 255)
        bands[2] = np.minimum(bands[2] * (mu_g / np.average(bands[2])), 255)
        return bands

    @staticmethod
    def _stretch(bands):
        for i, band in enumerate(bands):
            bands[i] = np.maximum(band - band.min(), 0)
        return bands

    @staticmethod
    def _max_white(bands):
        brightest = float(2 ** 8)
        for i, band in enumerate(bands):
            bands[i] = np.minimum(band * (brightest / float(band.max())), 255)
        return bands

    @staticmethod
    def _upper_scale(multi_bands, pan_band, multi_prof, pan_prof, sampling_method='bilinear'):

        if sampling_method == 'nearest':
            sampling_method = Resampling.nearest
        else:
            sampling_method = Resampling.bilinear

        if multi_bands.ndim == 2:
            upper_mul = np.empty(
                    (pan_band.shape[0], pan_band.shape[1]),
                    dtype = multi_bands.dtype
                    )
        else:
            upper_mul = np.empty(
                    (multi_bands.shape[0], pan_band.shape[0], pan_band.shape[1]),
                    dtype = multi_bands.dtype
                    )

        reproject(
                multi_bands,
                upper_mul,
                src_transform=multi_prof['transform'],
                src_crs=multi_prof['crs'],
                dst_transform=pan_prof['transform'],
                dst_crs=pan_prof['crs'],
                resampling=sampling_method
                )

        return upper_mul

    @staticmethod
    def _rescale(band, in_range=(0,65535), out_range=(0,255), dtype=np.uint8):

        return rescale_intensity(band, in_range, out_range).astype(dtype)


    @staticmethod
    def calc_brovey_dnf(bands, pan, rgb, rweight=1, gweight=1, bweight=0.2, irweight=0.1):

        """
        Pansharpen with Brovey method.
        DNF = (PAN - WI * IR) / (WR * R + WG * G + WB * B)
        pansharpened = multi_band * DNF
        """

        r = bands[rgb[0]]
        g = bands[rgb[1]]
        b = bands[rgb[2]]
        if len(rgb) == 4:
            ir = bands[rgb[3]]
            dnf_num = pan - ir * irweight
        else:
            ir = False
            dnf_num = pan

        dnf_denom = (r * rweight + g * gweight + b * bweight)/(rweight+gweight+bweight)
        dnf_mask = np.where(dnf_denom != 0, 1, 0).astype(np.bool)
        dnf_denom = np.where(dnf_denom == 0, 1, dnf_denom)
        dnf = np.nan_to_num(dnf_num / dnf_denom) * dnf_mask

        return dnf

    def Brovey(self):

        dnf = self.calc_brovey_dnf(self.multi_bands, self.pan_band, self.rgb)
        panshapend = np.empty(
                self.multi_bands.shape,
                dtype = np.float32
                )

        for i in range(self.multi_bands.shape[0]):
            panshapend[i] = self.multi_bands[i] * dnf

        return panshapend

