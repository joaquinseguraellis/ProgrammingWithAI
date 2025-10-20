# Libraries

import os
import cartopy
import cartopy.io.img_tiles
import itertools
import pyproj

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs

from pathlib import Path
from PIL import Image
from matplotlib.patheffects import Stroke, Normal
from matplotlib.patches import Rectangle

COUNTIES_SHP_PATH = Path('linea_de_limite_070110', 'linea_de_limite_070110.shp')
PROVINCES_SHP_PATH = Path('linea_de_limite_070111', 'linea_de_limite_070111.shp')
COUNTRIES_SHP_PATH = Path('argentina', 'argentina.shp')
DPI = 800

# Classes

class Map:

    def __init__(
            self, bbox = None,
            crs1 = ccrs.UTM(20, southern_hemisphere=True),
            crs2 = ccrs.PlateCarree(),
            fs: int = 14,
            tiler = cartopy.io.img_tiles.GoogleTiles(
                style="satellite", cache=True,
            ),
    ) -> None:
        self.bbox = bbox
        self.crs1 = crs1
        self.crs2 = crs2
        self.fs = fs
        self.tiler = tiler

    def figure(
            self,
            figsize: tuple[float, float] = None,
            dpi: float = 800,
            zoom = 12,
    ):
        self.fig = plt.figure(figsize=figsize, dpi=dpi)
        self.ax = self.fig.add_subplot(1, 1, 1, projection=self.crs1)
        self.ax.set_extent(self.bbox, crs=self.crs2)
        self.ax.add_image(self.tiler, zoom)
    
    def inset(self, bbox_ins, provinces = None, countries = None):
        axins = self.ax.inset_axes(
            [
                0.01,
                0.74,
                0.25 * (bbox_ins[1] - bbox_ins[0]) / (bbox_ins[3] - bbox_ins[2]),
                0.25,
            ],
            projection=self.crs2,
        )
        axins.coastlines()
        axins.stock_img()
        axins.set_extent(bbox_ins, crs=self.crs2)
        if provinces is not None:
            axins.add_geometries(
                provinces, self.crs2, edgecolor='black', alpha=0.1,
                facecolor='none', linewidth=0.5, linestyle=(2, (2, 2)),
            )
        if countries is not None:
            axins.add_geometries(
                countries, self.crs2, edgecolor='black', alpha=0.1,
                facecolor='none', linewidth=0.5, linestyle='-',
            )
        axins.add_artist(
            mpatches.Rectangle(
                (self.bbox[0], self.bbox[2]),
                self.bbox[1] - self.bbox[0],
                self.bbox[3] - self.bbox[2],
                color='red', ec="none", alpha=0.4,
            )
        )

# Functions

def reprojection(
        x, y, projparams = None,
        crs_from: pyproj.CRS = pyproj.CRS.from_epsg("4326"),
        crs_to: pyproj.CRS = None,
):
    """
    
    """
    if projparams is not None:
        crs_to = pyproj.Proj(
            f'+proj=sinu +R={projparams[0]} +nadgrids=@null +wktext'
        ).crs
    transformer = pyproj.Transformer.from_crs(
        crs_from, crs_to, always_xy=True,
    )
    return transformer.transform(x, y)

def scale_bar(
        self, n=4, loc='lower right', fraction=0.5,
        colors=["k", "w"], textcolor='black',
        crs_from: pyproj.CRS = pyproj.CRS.from_epsg("4326"),
        crs_to: pyproj.CRS = pyproj.CRS.from_epsg("32720"),
        hpad=0.1, vpad=0.1, labelsize=12, lw=1, zorder=None,
        bc='white', bpad=0.1,
):
    def get_unit(length):
        units = ['m', 'km', 'Mm', 'Gm', 'Tm']
        unit_index = 0
        scale = 1
        while length > 1000 and unit_index < len(units) - 1:
            length /= 1000
            scale /= 1000
            unit_index += 1
        s = str(length)
        li = s.find('.')
        li = 10**(li-1)
        li = round(length / li) * li
        return units[unit_index], li / scale, scale
    xlim, ylim = self.get_xlim(), self.get_ylim()
    xlim_m, _ = reprojection(xlim, ylim, crs_from=crs_from, crs_to=crs_to)
    width, height = xlim[1] - xlim[0], ylim[1] - ylim[0]
    length = fraction * width
    length_scale = width / (xlim_m[1] - xlim_m[0])
    unit, length_m, scale = get_unit(length / length_scale)
    length = length_m * length_scale
    hpad = hpad * width
    vpad = vpad * height
    if loc == 'lower right':
        x0, y0 = xlim[1] - length - hpad, ylim[0] + vpad
    elif loc == 'lower left':
        x0, y0 = xlim[0] + hpad, ylim[0] + vpad
    elif loc == 'upper right':
        x0, y0 = xlim[1] - length - hpad, ylim[1] - vpad
    elif loc == 'upper left':
        x0, y0 = xlim[0] + hpad, ylim[1] - vpad
    else:
        raise ValueError("Unsupported location for scale bar.")
    w = length / n
    y1 = y0 + vpad * 0.1
    bws = itertools.cycle(colors)
    x_values = np.array([[i, i+w] for i in np.arange(x0, x0+length, w)])
    y_values = np.array([[y0, y0] for _ in np.arange(0, length, w)])
    texts = [
        self.text(
            x0, y1, 0, ha='center', va='bottom',
            fontsize=labelsize, color=textcolor,
        )
    ]
    for i in np.arange(w, length*1.001, w):
        bbox = texts[-1].get_window_extent(
            self.figure.canvas.get_renderer()
        )
        bbox = self.transData.inverted().transform(bbox)
        if x0 + i > bbox[1, 0]:
            label = np.round(i * scale / length_scale, 1)
            if label == int(label):
                label = int(label)
            texts.append(
                self.text(
                    x0 + i, y1, label, ha='center', va='bottom',
                    fontsize=labelsize, color=textcolor,
                )
            )
    bbox = texts[-1].get_window_extent(
        self.figure.canvas.get_renderer()
    )
    bbox = self.transData.inverted().transform(bbox)
    texts.append(
        self.text(
            bbox[1, 0] + 0.2*w, y1, unit,
            ha='left', va='bottom',
            fontsize=labelsize, color=textcolor,
        )
    )
    bbox = texts[-1].get_window_extent(
        self.figure.canvas.get_renderer()
    )
    bbox = self.transData.inverted().transform(bbox)
    x1 = bbox[1, 0]
    y1 = bbox[1, 1]
    offset = (bbox[1, 0] - bbox[0, 0]) * 1.3
    x_values -= offset
    for t in texts:
        t.set_x(t.get_position()[0] - offset)
    for xx, yy in zip(x_values, y_values):
        bw = next(bws)
        self.plot(
            xx, yy, color=bw, linewidth=lw,
            clip_on=False, zorder=zorder, solid_capstyle='butt',
            path_effects=[
                Stroke(linewidth=1.2*lw, foreground="black"),
                Normal(),
            ],
        )
    pad = bpad * width
    rect = Rectangle(
        (x0 - offset - pad, y0 - pad),
        x1 - x0 + 2*pad, y1 - y0 + 2*pad,
        ec="none", fc=bc, alpha=0.2,
    )
    self.add_patch(rect)

def north_arrow(self, path, position):
    img = Image.open(path)
    img = np.array(img)
    ax_ = self.add_axes(position)
    ax_.imshow(img)
    ax_.axis('off')

def zebra_frame(
        self, bbox, dx, dy, fs=12, lw=2, crs=None, zorder=None,
        degrees=True,
        xlabels=None, xticks=None, xlocs=None,
        ylabels=None, yticks=None, ylocs=None,
):
    def get_ticks_locs(d, lims, ticks=None, locs=None):
        dec = int(np.ceil(np.log10(1 / d)))
        low, high = round(lims[0], dec), round(lims[1], dec)
        if low > lims[0]:
            low -= d
        if high < lims[1]:
            high += d
        if ticks is None:
            ticks = np.arange(low, high + 0.01*d, 0.25*d)
        ticks = ticks[ticks > lims[0]]
        ticks = ticks[ticks < lims[1]]
        ticks = np.concatenate([lims[:1], ticks, lims[1:]])
        if locs is None:
            locs = np.arange(low, high + 0.01*d, 0.5*d)
            locs = locs[locs > lims[0]]
            locs = locs[locs < lims[1]]
        return ticks, locs
    xticks, xlocs = get_ticks_locs(dx, bbox[:2], xticks, xlocs)
    yticks, ylocs = get_ticks_locs(dy, bbox[2:], yticks, ylocs)
    if degrees:
        gl = self.gridlines(
            draw_labels=True, dms=False, linestyle='-',
            xlocs=xlocs, ylocs=ylocs, color='black', lw=0.1*lw,
        )
        gl.xlabel_style = {'size': fs}
        gl.ylabel_style = {'size': fs, 'rotation': 90}
    else:
        for xx in xlocs:
            self.plot(
                [xx, xx], [yticks[0], yticks[-1]],
                lw=lw*0.2, color='black',
            )
        for yy in ylocs:
            self.plot(
                [xticks[0], xticks[-1]], [yy, yy],
                lw=lw*0.2, color='black',
            )
        self.set_xticks(xlocs)
        self.set_yticks(ylocs)
    if xlabels is not None:
        self.set_xticklabels(xlabels, fontsize=fs)
    else:
        self.set_xticklabels(xlocs, fontsize=fs)
    if ylabels is not None:
        self.set_yticklabels(ylabels, fontsize=fs)
    else:
        self.set_yticklabels(ylocs, fontsize=fs)
    self.spines["geo"].set_visible(False)
    for axis in ['x', 'y']:
        for bc in bbox:
            bws = itertools.cycle(["k", "w"])
            if axis == 'x':
                x_values = np.column_stack((xticks[:-1], xticks[1:]))
                y_values = np.full(x_values.shape, bc)
            else:
                y_values = np.column_stack((yticks[:-1], yticks[1:]))
                x_values = np.full(y_values.shape, bc)
            for idx, (xx, yy) in enumerate(zip(x_values, y_values)):
                capstyle = "butt" if idx not in (0, x_values.shape[0] - 1) else "projecting"
                bw = next(bws)
                self.plot(
                    xx, yy, color=bw, linewidth=lw,
                    clip_on=False, transform=crs,
                    zorder=zorder, solid_capstyle=capstyle,
                    path_effects=[
                        Stroke(linewidth=1.2*lw, foreground="black"),
                        Normal(),
                    ],
                )