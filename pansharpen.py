#!/usr/bin/env python3

import rasterio as rio
import numpy as np
from rasterio.warp import reproject, Resampling
from skimage.exposure import rescale_intensity

class LANDSAT8_PRE(object):
    def __init__(self, landsat8_id, landsat8_base='https://s3-us-west-2.amazonaws.com/landsat-pds/c1/L8/{}/{}/{}/{}'):

        pathrow = landsat8_id.split('_')[2]
        self.sceneid = landsat8_id
        self.path = pathrow[:3]
        self.row = pathrow[3:]
        self.rgb = [2,1,0,3]
        self.landsat8_base = landsat8_base
        bands = []
        multi_prof = None
        for band in range(2,6):
            band_image, band_profile = self._get_band(band)
            bands.append(band_image)
            if not multi_prof:
                multi_prof = band_profile

        self.multi_prof = multi_prof
        mul_shape = bands[-1].shape
        self.multi_bands = np.zeros((len(bands), mul_shape[0], mul_shape[1]), dtype = bands[-1].dtype)
        for i in range(len(bands)):
            self.multi_bands[i] = bands[i]

        del bands

        self.pan_band, self.pan_prof = self._get_band(8)
        self.qa_band, self.qa_prof = self._get_band('QA')
        self.cloud_mask, self.snow_mask, self.fill_mask = self._get_mask()

        return

    def _get_band(self, bandname):

        url_base = self.landsat8_base
        filename = '{}_B{}.TIF'.format(self.sceneid, bandname)
        download_url = url_base.format(self.path,self.row,self.sceneid,filename)
        with rio.open(download_url) as src:
            band = src.read(1)
            band_profile = src.profile

        return band, band_profile

    def _get_mask(self):

        cloud_high_conf = int('0000000001100000',2)
        snow_high_conf = int('0000011000000000',2)
        fill_pixels = int('0000000000000001', 2)
        cloud_mask = np.bitwise_and(self.qa_band, cloud_high_conf) == cloud_high_conf
        snow_mask = np.bitwise_and(self.qa_band, snow_high_conf) == snow_high_conf
        fill_mask = np.bitwise_and(self.qa_band, fill_pixels) == fill_pixels

        return cloud_mask, snow_mask, fill_mask

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

        #self.multi_bands = Pansharpen._upper_scale(self.multi_bands, self.pan_band, self.multi_prof, self.pan_prof)
        self.multi_bands = multi
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

    @staticmethod
    def rescale_8(band, cut=(0.1,99)):
        return Pansharpen.rescale_band(band, cut)

    @staticmethod
    def rescale_16(band, cut=(0.1,99)):
        return Pansharpen.rescale_band(band, cut, 'uint16')

    @staticmethod
    def rescale_band(band, cut=(0,100), dtype='uint8'):
        low, high = np.percentile(band[band>0], cut)
        return rescale_intensity(band, in_range=(low,high), out_range=(0,np.iinfo(dtype).max)).astype(dtype)

    @staticmethod
    def bgr2hsv(bgr):
        """
        h = 0 ~ 360 degree
        s = 0.0 ~ 1.0
        v = maximum( depend on input dtype(uint8 max 255, uint16 max 65535)
        """
        r = bgr[2]
        g = bgr[1]
        b = bgr[0]
        width, height = r.shape
        s = np.zeros(r.shape, dtype=np.float32)
        h = np.zeros(r.shape, dtype=np.float32)
        mm = np.maximum(np.maximum(r,g),b)
        mn = np.minimum(np.minimum(r,g),b)
        delta = mm - mn
        s[mm>0] = delta[mm>0]/mm[mm>0]
        s[mm==0] = 0
        delta[delta == 0] = 1
        rt = r/delta
        gt = g/delta
        bt = b/delta
        h = 240 + 60*(rt - gt)
        h[g == mm] = 120 + 60 * (bt[g == mm] - rt[g == mm])
        h[r == mm] = 60 * (gt[r == mm] - bt[r == mm])
        h[mm == mn] = 0
        h[h<0] += 360.0
        v = mm.astype(np.float32)
        return np.stack((h,s,v))

    @staticmethod
    def hsv2bgr(hsv,dtype='uint16'):
        """
        h = 0 ~ 360 degree
        s = 0.0 ~ 1.0
        v = maximum( depend on input dtype(uint8 max 255, uint16 max 65535)
        """
        h = hsv[0]
        s = hsv[1]
        v = hsv[2].astype(dtype)
        width,height = v.shape
        v = v.flatten()
        s = s.flatten()
        h = h.flatten()
        i = np.int8(h/60)
        mn = v * (1-s)
        mx = v
        delta = mx - mn
        r , g , b = np.zeros(v.shape, dtype=dtype),np.zeros(v.shape, dtype=dtype),np.zeros(v.shape, dtype=dtype)
        r[i==0] = mx[i==0]
        g[i==0] = (h[i==0]/60)*delta[i==0]+mn[i==0]
        b[i==0] = mn[i==0]

        r[i==1] = ((120-h[i==1])/60)*delta[i==1]+mn[i==1]
        g[i==1] = mx[i==1]
        b[i==1] = mn[i==1]

        r[i==2] = mn[i==2]
        g[i==2] = mx[i==2]
        b[i==2] = ((h[i==2]-120)/60)*delta[i==2]+mn[i==2]

        r[i==3] = mn[i==3]
        g[i==3] = ((240-h[i==3])/60)*delta[i==3]+mn[i==3]
        b[i==3] = mx[i==3]

        r[i==4] = ((h[i==4]-240)/60)*delta[i==4]+mn[i==4]
        g[i==4] = mn[i==4]
        b[i==4] = mx[i==4]

        r[i==5] = mx[i==5]
        g[i==5] = mn[i==5]
        b[i==5] = ((360-h[i==5])/60)*delta[i==5]+mn[i==5]

        r[s==0] = mx[s==0]
        g[s==0] = mx[s==0]
        b[s==0] = mx[s==0]

        return np.stack((b.reshape(width,height),g.reshape(width,height),r.reshape(width,height)))

    def Brovey(self):

        multi_bands = Pansharpen._upper_scale(self.multi_bands, self.pan_band, self.multi_prof, self.pan_prof)

        dnf = self.calc_brovey_dnf(multi_bands, self.pan_band, self.rgb)
        panshapend = np.empty(
                multi_bands.shape,
                dtype = np.float32
                )

        for i in range(multi_bands.shape[0]):
            panshapend[i] = multi_bands[i] * dnf

        return panshapend

    def HSV(self, hsv=False):
        import cv2

        bgr = np.stack((self.multi_bands[self.rgb[0]],self.multi_bands[self.rgb[1]],self.multi_bands[self.rgb[2]]))

        hsv = Pansharpen.bgr2hsv(bgr)
        
        newhsv = Pansharpen._upper_scale(hsv, self.pan_band, self.multi_prof, self.pan_prof)
        return newhsv

        """
        rgb = cv2.merge((b,g,r))
        hsv = cv2.cvtColor(rgb,cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv)
        hs = np.stack((h,s))
        newhs = Pansharpen._upper_scale(hs, self.pan_band, self.multi_prof, self.pan_prof)
        h = newhs[0]
        s = newhs[1]
        v = Pansharpen.rescale_8(self.pan_band)
        hsv = cv2.merge((h,s,v))
        if hsv:
            return hsv
        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

        return bgr
        """
