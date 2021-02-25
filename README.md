# Pansharpen 実験場

## はじめに

このpythonスクリプトはpansharpenが何かを理解するために実験的に作成されています。

主にtellus jupyter開発環境で使えるようにすることを目標にしていますが、なにせメモリをいっぱい食べる仕様なので、省メモリ化をしていかないとだめかなあと思っております。

## 使い方

landsat8はawsから取得することを想定しています。band2,3,4,5,8,QAを読み込みnumpyに格納するLANDSAT8_PREクラスがあります。

```
from pansarpen import *

d = LANDSAT8_PRE('LC08_L1TP_108036_20181125_20181210_01_T1')
```

読み込み後、pansharpenを呼び出します。
読み込むと、マルチバンド画像をbilinearでpanと同じサイズにしてpansharpenの準備が完了します。

```
p = Pansharpen(d.multi_bands, d.multi_prof, d.pan_band, d_pan_prof, d.rgb)
```

現在はBrovey法しかありません。

```
brovey = p.Brovey()
```

broveyにはr,g,b,irのpansharpenされたnumpy.float32の配列が格納されます。
このままTIFファイルにするには

```
import rasterio as rio

kwargs = d_pan_prof
kwargs.update({
	'dtype', brovey.dtype,
	'compress': 'lzw',
	'count': brovey.shape[0]
	})

with rio.open('pansharpend.tif','w', **kwargs) as pan:
	pan.write(brovey)
```

PNGにするのは、色々方法がありますが、

```
def rescale_8(band, cut=(0,100)):
	from skimage.exposure import rescale_intensity
	low, high = np.percentile(band[band>0], cut)
	return rescale_intensity(band, in_range=(low,high), out_range=(0,255)).astype(np.uint8)

r = rescale_8(brovey[2], (0.1,99))
g = rescale_8(brovey[1], (0.1,99))
b = rescale_8(brovey[0], (0.1,99))
ir = rescale_8(brovey[3], (0.1,99))

import cv2
bgr = cv2.merge((b,g,r))
cv2.imsave('rgb_pansharpen.png', bgr)

vegi = cv2.merge((g,r,ir))
cv2.imsave('543_pansharpen.png', vegi)
```

Enjoy!
