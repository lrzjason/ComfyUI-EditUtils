"""BreastSizeControl_EditUtils canvas node."""

import math
import torch
import json

# Payload written by web/js/breast_size_control.js.  Geometry is normalised so
# the drawing survives a canvas resize: cx/rx are fractions of the width,
# cy/ry fractions of the height.  This default is only a starting point -- the
# canvas owns the value from the moment the node is created.
#
# The default drawing is a circle pair totalling the measured "medium" class
# area (3.22%, so 1.61% each) on the default 832x1216 canvas:
#   rx = sqrt(0.0161 * 1216 / (pi * 832)) = 0.086545
#   ry = rx * 832 / 1216                  = 0.059215   -> both are 72.005 px
# Because radii are normalised per axis those two numbers are a circle only at
# that aspect; the canvas re-derives ry whenever the canvas aspect changes, and
# every circle it creates itself is aspect correct.
DEFAULT_BREAST_MARKERS = json.dumps(
    {
        "version": 1,
        "color": "#FADBAE",
        "circles": [
            {"cx": 0.355, "cy": 0.46, "rx": 0.086545, "ry": 0.059215, "angle": 0},
            {"cx": 0.645, "cy": 0.46, "rx": 0.086545, "ry": 0.059215, "angle": 0},
        ],
    },
    separators=(",", ":"),
)


def _hex_color_to_float01(text, fallback=(250, 219, 174)):
    """``'#FADBAE'`` -> ``(0.980, 0.859, 0.682)``.

    Accepts 3, 6 or 8 hex digits (8-digit alpha is ignored) with or without the
    leading ``#``.  Never raises: anything unparseable returns ``fallback``,
    which is ``#FADBAE`` in 0-1 floats.
    """
    if isinstance(text, str):
        digits = text.strip().lstrip("#").strip()
        if len(digits) in (3, 4):
            digits = "".join(ch * 2 for ch in digits)
        if len(digits) in (6, 8):
            try:
                return tuple(int(digits[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
            except ValueError:
                pass
    return tuple(c / 255.0 for c in fallback)


class BreastSizeControl_EditUtils:
    """Author a breast-size control (condition) image on the node's canvas.

    The output is a black RGB image carrying one or two filled markers -- the
    same contract as the ``_bd`` condition images in
    ``E:\\images\\breasts_size_control``: black background, marker fill
    ``#FADBAE`` by default, the marker's **area** standing for the breast size
    and its **position** for where the breasts sit.  Markers are drawn as
    ellipses, so a marker can be a true circle (corner handle / wheel / Shift)
    or an anisotropic ellipse (edge handles), which is what the measured
    ``_bd`` images actually use (median bbox aspect 0.91).

    All geometry lives in the ``markers`` JSON payload owned by the canvas UI::

        {"version": 1,
         "color": "#FADBAE",
         "circles": [{"cx": 0.355, "cy": 0.46, "rx": 0.135, "ry": 0.135,
                      "angle": 0.0}, ...]}

    ``cx``/``rx`` are fractions of the width, ``cy``/``ry`` fractions of the
    height, so the drawing is resolution independent.  ``angle`` is degrees,
    positive = clockwise on screen (same convention as ``krea2_geom_lab``); a
    rotation moves the marker without changing its area, so it never disturbs
    the size class.

    Connect ``image`` to force the output to that image's exact pixel size --
    the condition image must match its target pixel for pixel, otherwise the
    marker geometry lands in the wrong place once the pair is bucketed.  With
    nothing connected the ``width``/``height`` widgets define the canvas.

    ``image`` is used for its dimensions **only**.  The output is always a fresh
    black canvas carrying the markers: none of the connected image's pixels ever
    reach it, so the result is exactly two colours (black + marker fill).
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {
                    "default": 832, "min": 16, "max": 16384, "step": 8,
                    "tooltip": "Canvas width, used when no image is connected. The drawing is "
                               "stored normalised, so changing this only rescales the output.",
                }),
                "height": ("INT", {
                    "default": 1216, "min": 16, "max": 16384, "step": 8,
                    "tooltip": "Canvas height, used when no image is connected.",
                }),
                "markers": ("STRING", {
                    "default": DEFAULT_BREAST_MARKERS,
                    "multiline": False,
                    "socketless": True,
                    "advanced": True,
                    "tooltip": "Owned by the node's canvas (JSON: color + one or two circles with "
                               "cx/cy/rx/ry/angle). Edit the markers by dragging on the canvas, "
                               "not by hand.",
                }),
                "antialias": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "False reproduces the existing _bd images exactly (two colours, "
                               "hard edges); True supersamples the marker edges 4x4.",
                }),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "Sets the output's exact pixel size (batch: first frame decides). "
                               "Used for its size only: the output is always a black canvas plus the "
                               "markers, never the connected image. Required for the _bd contract "
                               "that a condition image equals its target's size.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "render_condition"
    CATEGORY = "image"
    DESCRIPTION = (
        "Draw one or two breast-size control markers (black background, #FADBAE fill) directly "
        "on the node's canvas. Drag a marker to move it, drag its edge handles to shape it "
        "(circle or ellipse), the corner handles / wheel to scale it, the rotation handle to "
        "angle it, and use the size-class preset buttons to set the measured "
        "flat/small/medium/large/huge/gigantic areas."
    )

    @staticmethod
    def _decode_markers(raw):
        """Parse the canvas payload -> ``(color01, [(cx, cy, rx, ry, angle)])``.

        Deliberately forgiving: a malformed or empty payload renders a plain
        black canvas instead of failing the whole workflow.
        """
        data = {}
        if isinstance(raw, str) and raw.strip():
            try:
                data = json.loads(raw)
            except ValueError:
                data = {}
        if not isinstance(data, dict):
            data = {}

        color = _hex_color_to_float01(data.get("color"))

        circles = []
        items = data.get("circles")
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                try:
                    cx = float(item.get("cx", 0.5))
                    cy = float(item.get("cy", 0.5))
                    rx = float(item.get("rx", 0.0))
                    ry = float(item.get("ry", 0.0))
                except (TypeError, ValueError):
                    continue
                try:
                    # A broken angle must not discard an otherwise valid marker.
                    angle = float(item.get("angle") or 0.0)
                except (TypeError, ValueError):
                    angle = 0.0
                if not all(math.isfinite(v) for v in (cx, cy, rx, ry, angle)):
                    continue
                if rx <= 0.0 or ry <= 0.0:
                    continue
                circles.append((cx, cy, rx, ry, angle))
        return color, circles

    @staticmethod
    def _ellipse_coverage(cx, cy, rx, ry, angle_deg, height, width, samples):
        """Antialiased coverage of one ellipse, restricted to its bounding box.

        ``angle_deg`` rotates the marker clockwise on screen about its centre,
        which only changes the sampled region -- the area is unaffected.
        Returns ``(coverage, (y0, x0, y1, x1))`` with coverage shaped
        ``(y1 - y0, x1 - x0)``, or ``None`` when the ellipse is off-canvas.
        ``samples == 1`` samples pixel centres, giving a hard-edged 0/1 mask.
        """
        theta = math.radians(angle_deg)
        cos_t, sin_t = math.cos(theta), math.sin(theta)

        # A rotated ellipse no longer fits the axis-aligned rx/ry box: its
        # half-extents are the projections of both axes.
        half_w = math.sqrt((rx * cos_t) ** 2 + (ry * sin_t) ** 2)
        half_h = math.sqrt((rx * sin_t) ** 2 + (ry * cos_t) ** 2)

        x0 = max(0, int(math.floor(cx - half_w)))
        x1 = min(width, int(math.ceil(cx + half_w)) + 1)
        y0 = max(0, int(math.floor(cy - half_h)))
        y1 = min(height, int(math.ceil(cy + half_h)) + 1)
        if x1 <= x0 or y1 <= y0:
            return None

        nx, ny = x1 - x0, y1 - y0
        ss = max(1, int(samples))
        # Sub-pixel sample centres, in pixel coordinates.
        sx = (torch.arange(nx * ss, dtype=torch.float32) + 0.5) / ss + x0
        sy = (torch.arange(ny * ss, dtype=torch.float32) + 0.5) / ss + y0
        dx = (sx - cx).view(1, -1)
        dy = (sy - cy).view(-1, 1)
        # Rotate the samples into the marker's local frame (inverse rotation).
        lx = dx * cos_t + dy * sin_t
        ly = -dx * sin_t + dy * cos_t
        inside = (lx / rx) ** 2 + (ly / ry) ** 2 <= 1.0
        cov = inside.to(torch.float32).view(ny, ss, nx, ss).mean(dim=(1, 3))
        return cov, (y0, x0, y1, x1)

    def render_condition(self, width, height, markers, antialias=False, image=None):
        if image is not None:
            if image.ndim != 4:
                raise ValueError(
                    f"image must be a batched IMAGE (B, H, W, C); got shape {tuple(image.shape)}"
                )
            height, width = int(image.shape[1]), int(image.shape[2])

        width = max(1, int(width))
        height = max(1, int(height))

        color, circles = self._decode_markers(markers)
        canvas = torch.zeros((height, width, 3), dtype=torch.float32)
        color_t = torch.tensor(color, dtype=torch.float32).view(1, 1, 3)
        samples = 4 if antialias else 1

        for cx_n, cy_n, rx_n, ry_n, angle in circles:
            coverage = self._ellipse_coverage(
                cx_n * width, cy_n * height, rx_n * width, ry_n * height,
                angle, height, width, samples,
            )
            if coverage is None:
                continue
            cov, (y0, x0, y1, x1) = coverage
            cov = cov.unsqueeze(-1)
            # Overlapping markers merge (union of coverage), they never add up.
            region = canvas[y0:y1, x0:x1]
            canvas[y0:y1, x0:x1] = region * (1.0 - cov) + color_t * cov

        return {
            # ComfyUI flattens every `ui` value (`for y in x[k]` in
            # execution.get_output_from_returns), so each one must be a list --
            # a bare int raises "TypeError: 'int' object is not iterable".
            "ui": {"width": [width], "height": [height]},
            "result": (canvas.unsqueeze(0),),
        }
