/**
 * ComfyUI-EditUtils -- BreastSizeControl_EditUtils canvas editor.
 *
 * A WYSIWYG authoring canvas for breast-size control (condition) images: the
 * black background and the filled markers you see on the canvas ARE the pixels
 * the Python node renders (ctx.ellipse and the node's rasterizer use the same
 * inside test).  One or two markers; a marker's area is the size dial and its
 * position is where the breasts sit.
 *
 * Tools
 *   left drag empty space   create a marker (max 2)
 *   left drag the body      move
 *   left drag N/S/E/W edge  change ry / rx on its own  -> circle becomes ellipse
 *   shift + edge drag       resize uniformly (stays a circle)
 *   left drag a corner      resize uniformly
 *   wheel over a marker     resize uniformly
 *   right click a marker    delete
 *   preset buttons          set the measured flat..gigantic areas
 *   colour picker           marker fill, #FADBAE by default
 *
 * The payload the node receives lives in the hidden `markers` widget:
 *   {"version":1,"color":"#FADBAE",
 *    "circles":[{"cx":..,"cy":..,"rx":..,"ry":..}]}
 * cx/rx are fractions of the width, cy/ry fractions of the height, so the
 * drawing is resolution independent and matches the node's renderer exactly.
 * Note that the normalised form makes the area fraction simply `pi * rx * ry`.
 *
 * Geometry and layout follow ComfyUI-KJNodes' canvas editors (BaseEditorCanvas /
 * PointsEditor) -- the pattern the user pointed at in E:\ComfyUI -- but this
 * file is self-contained and imports nothing from other custom nodes.
 */

const NODE_NAME = "BreastSizeControl_EditUtils";

const COLOR_DEFAULT = "#FADBAE";
const MAX_MARKERS = 2;
const MARGIN = 14;
const MAX_DISPLAY = 1024;
const MIN_RADIUS_PX = 3;      // a marker never gets smaller than this on screen
const MAX_RADIUS_NORM = 2.0;  // nor larger than twice the canvas, as a fraction
const ROTATE_ARM = 20;        // canvas px between the ellipse's top and the rotation handle

// Measured per-class median marker area fraction (analysis.md appendix A).
const PRESETS = [
  { name: "flat", area: 0.0149 },
  { name: "small", area: 0.0294 },
  { name: "medium", area: 0.0322 },
  { name: "large", area: 0.0718 },
  { name: "huge", area: 0.1395 },
  { name: "gigantic", area: 0.1650 },
];

// Fallback only; nodes.py owns the real default once the widget exists.
// A circle pair totalling the measured "medium" class area (3.22%, i.e. 1.61%
// each) on the widget's default 832x1216 canvas:
//   rx = sqrt(0.0161 * 1216 / (pi * 832)) = 0.086545, ry = rx * 832 / 1216 = 0.059215
// so rx * 832 == ry * 1216 == 72.005 px.
const DEFAULT_PAYLOAD = {
  version: 1,
  color: COLOR_DEFAULT,
  circles: [
    { cx: 0.355, cy: 0.46, rx: 0.086545, ry: 0.059215, angle: 0 },
    { cx: 0.645, cy: 0.46, rx: 0.086545, ry: 0.059215, angle: 0 },
  ],
};

function clone(obj) {
  return JSON.parse(JSON.stringify(obj));
}

function chainCallback(object, property, callback) {
  if (!object) return;
  const original = object[property];
  if (typeof original === "function") {
    object[property] = function (...args) {
      const result = original.apply(this, args);
      callback.apply(this, args);
      return result;
    };
  } else {
    object[property] = callback;
  }
}

function clamp(value, lo, hi) {
  return Math.max(lo, Math.min(hi, value));
}

/** Degrees -> (-180, 180].  Positive is clockwise on screen (y grows down). */
function normalizeAngle(deg) {
  if (!Number.isFinite(deg)) return 0;
  return ((((deg + 180) % 360) + 360) % 360) - 180;
}

/** Local (unrotated) coordinates of a canvas-space point relative to a marker. */
function toLocal(dx, dy, radians) {
  const c = Math.cos(radians);
  const s = Math.sin(radians);
  return { x: dx * c + dy * s, y: -dx * s + dy * c };
}

class BreastSizeEditor {
  constructor(node) {
    this.node = node;
    this.canvas = document.createElement("canvas");
    this.ctx = this.canvas.getContext("2d");
    this.markers = [];
    this.color = COLOR_DEFAULT;
    this.selected = 0;
    this.guideImg = null;
    this.guideEnabled = true;
    this.drag = null;

    this.node._bscCanvasH = 480;
    this.node._bscControlsH = 90;

    this._readWidgets();
    this._buildDom();
    this._readState();
    this._relayout();
    // Write the payload back once so it is canonical from the start: every
    // marker spelled out with cx/cy/rx/ry/angle, no stale keys.
    this._commit();
    requestAnimationFrame(() => {
      this._measureControls(true);
      this._relayout();
      this._syncFromConnectedImage();
    });
  }

  // ── widgets & DOM ────────────────────────────────────────────────────────

  _readWidgets() {
    const find = (name) => (this.node.widgets || []).find((w) => w.name === name);
    this.widthWidget = find("width");
    this.heightWidget = find("height");
    this.markersWidget = find("markers");
    // The payload is the drawing itself; it is edited on the canvas, never by hand.
    if (this.markersWidget) {
      this.markersWidget.hidden = true;
      this.markersWidget.options = this.markersWidget.options || {};
      this.markersWidget.options.hidden = true;
    }
  }

  _buildDom() {
    const canvasWrap = document.createElement("div");
    canvasWrap.style.cssText =
      "position:relative;display:flex;justify-content:center;align-items:center;" +
      "width:100%;box-sizing:border-box;overflow:hidden;";
    canvasWrap.appendChild(this.canvas);
    this.canvas.style.cssText = "display:block;touch-action:none;cursor:default;";

    this.controls = document.createElement("div");
    this.controls.style.cssText =
      "display:flex;flex-wrap:wrap;gap:4px;align-items:center;width:100%;" +
      "box-sizing:border-box;padding:2px 0;font:11px sans-serif;color:#bbb;";

    this._addRow(this._buildActionRow());
    this._addRow(this._buildPresetRow());
    this._addRow(this._buildRotateRow());
    this.status = document.createElement("span");
    this.status.style.cssText = "width:100%;font:11px sans-serif;color:#8ab4f8;";
    this.controls.appendChild(this.status);

    this.node._bscCanvasWidget = this.node.addDOMWidget(
      "bsc_canvas",
      "BSCEditorCanvas",
      canvasWrap,
      {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => this.node._bscCanvasH,
        getMaxHeight: () => this.node._bscCanvasH,
        getHeight: () => this.node._bscCanvasH,
      },
    );
    this.node._bscControlsWidget = this.node.addDOMWidget(
      "bsc_controls",
      "BSCEditorControls",
      this.controls,
      {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => this.node._bscControlsH,
        getMaxHeight: () => this.node._bscControlsH,
        getHeight: () => this.node._bscControlsH,
      },
    );

    this._setupEvents();
  }

  _addRow(element) {
    this.controls.appendChild(element);
  }

  _button(label, title, onClick) {
    const button = document.createElement("button");
    button.textContent = label;
    button.title = title || "";
    button.style.cssText =
      "height:22px;padding:0 7px;font:11px sans-serif;cursor:pointer;" +
      "background:#2b2b2b;color:#ddd;border:1px solid #555;border-radius:3px;";
    button.addEventListener("pointerdown", (e) => e.stopPropagation());
    button.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      onClick();
    });
    return button;
  }

  _buildActionRow() {
    const row = document.createElement("div");
    row.style.cssText = "display:flex;flex-wrap:wrap;gap:4px;align-items:center;width:100%;";

    row.appendChild(this._button("+ Marker", "Add a marker", () => this._addMarkerAt()));
    row.appendChild(this._button("Clear", "Remove all markers", () => {
      this.markers = [];
      this.selected = 0;
      this._commit();
    }));

    const colourLabel = document.createElement("label");
    colourLabel.style.cssText = "display:flex;align-items:center;gap:4px;";
    colourLabel.title = "Marker fill colour (default #FADBAE)";
    colourLabel.appendChild(document.createTextNode("fill"));
    this.colorInput = document.createElement("input");
    this.colorInput.type = "color";
    this.colorInput.value = COLOR_DEFAULT;
    this.colorInput.style.cssText = "width:32px;height:22px;padding:0;border:1px solid #555;background:#2b2b2b;cursor:pointer;";
    this.colorInput.addEventListener("pointerdown", (e) => e.stopPropagation());
    this.colorInput.addEventListener("input", () => {
      this.color = this.colorInput.value.toUpperCase();
      this._commit();
    });
    colourLabel.appendChild(this.colorInput);
    row.appendChild(colourLabel);

    const guideLabel = document.createElement("label");
    guideLabel.style.cssText = "display:flex;align-items:center;gap:3px;";
    guideLabel.title = "Show the connected image as a dimmed underlay while drawing (never part of the output)";
    this.guideInput = document.createElement("input");
    this.guideInput.type = "checkbox";
    this.guideInput.checked = true;
    this.guideInput.style.cssText = "margin:0;";
    this.guideInput.addEventListener("pointerdown", (e) => e.stopPropagation());
    this.guideInput.addEventListener("change", () => {
      this.guideEnabled = this.guideInput.checked;
      this._render();
    });
    guideLabel.appendChild(this.guideInput);
    guideLabel.appendChild(document.createTextNode("guide"));
    row.appendChild(guideLabel);

    return row;
  }

  _buildPresetRow() {
    const row = document.createElement("div");
    row.style.cssText = "display:flex;flex-wrap:wrap;gap:3px;align-items:center;width:100%;";
    for (const preset of PRESETS) {
      const label = `${preset.name} ${(preset.area * 100).toFixed(1)}%`;
      row.appendChild(this._button(
        label,
        `Set the selected marker (or all markers when none is selected) to the measured ` +
        `${preset.name} area, ${(preset.area * 100).toFixed(2)}% of the frame`,
        () => this._applyPreset(preset),
      ));
    }
    return row;
  }

  /** Rotation controls: fine nudges plus a live readout of the angle. */
  _buildRotateRow() {
    const row = document.createElement("div");
    row.style.cssText = "display:flex;flex-wrap:wrap;gap:3px;align-items:center;width:100%;";

    row.appendChild(this._button("↺ 15°", "Rotate the selected marker 15 degrees anticlockwise",
      () => this._rotateSelected(-15)));
    row.appendChild(this._button("↻ 15°", "Rotate the selected marker 15 degrees clockwise",
      () => this._rotateSelected(15)));

    this.angleInput = document.createElement("input");
    this.angleInput.type = "number";
    this.angleInput.step = "5";
    this.angleInput.value = "0";
    this.angleInput.title = "Marker angle in degrees (positive = clockwise on screen)";
    this.angleInput.style.cssText =
      "width:58px;height:20px;font:11px sans-serif;background:#2b2b2b;color:#ddd;" +
      "border:1px solid #555;border-radius:3px;padding:0 3px;";
    this.angleInput.addEventListener("pointerdown", (e) => e.stopPropagation());
    this.angleInput.addEventListener("change", () => {
      const marker = this.markers[this.selected];
      if (!marker) return;
      marker.angle = normalizeAngle(Number(this.angleInput.value) || 0);
      this._commit();
    });
    row.appendChild(this.angleInput);
    row.appendChild(document.createTextNode("°"));

    row.appendChild(this._button("0°", "Reset the selected marker's angle",
      () => this._setSelectedAngle(0)));

    return row;
  }

  _rotateSelected(delta) {
    const marker = this.markers[this.selected];
    if (!marker) return;
    marker.angle = normalizeAngle((marker.angle || 0) + delta);
    this._commit();
  }

  _setSelectedAngle(angle) {
    const marker = this.markers[this.selected];
    if (!marker) return;
    marker.angle = normalizeAngle(angle);
    this._commit();
  }

  _updateAngleReadout() {
    if (!this.angleInput) return;
    const marker = this.markers[this.selected];
    const value = marker ? normalizeAngle(marker.angle || 0) : 0;
    if (document.activeElement !== this.angleInput) {
      this.angleInput.value = `${Math.round(value * 10) / 10}`;
    }
  }

  // ── state <-> widgets ────────────────────────────────────────────────────

  _readState() {
    let data = null;
    const raw = this.markersWidget ? this.markersWidget.value : "";
    if (typeof raw === "string" && raw.trim()) {
      try {
        data = JSON.parse(raw);
      } catch (error) {
        console.warn("[BSC] unreadable markers payload, using the default", error);
      }
    }
    if (!data || typeof data !== "object") data = clone(DEFAULT_PAYLOAD);

    this.color = /^#[0-9a-fA-F]{6}$/.test(data.color || "") ? data.color.toUpperCase() : COLOR_DEFAULT;
    this.colorInput.value = this.color;

    this.markers = (Array.isArray(data.circles) ? data.circles : [])
      .map((c) => ({
        cx: +c.cx,
        cy: +c.cy,
        rx: +c.rx,
        ry: +c.ry,
        angle: normalizeAngle(+c.angle || 0),
      }))
      .filter((c) => [c.cx, c.cy, c.rx, c.ry].every(Number.isFinite) && c.rx > 0 && c.ry > 0)
      .slice(0, MAX_MARKERS);
    this.selected = this.markers.length ? 0 : -1;
  }

  /** Write the payload the Python node consumes. */
  _commit() {
    if (this.markersWidget) {
      this.markersWidget.value = JSON.stringify({
        version: 1,
        color: this.color,
        circles: this.markers.map((m) => ({
          cx: +m.cx.toFixed(6),
          cy: +m.cy.toFixed(6),
          rx: +m.rx.toFixed(6),
          ry: +m.ry.toFixed(6),
          angle: +normalizeAngle(m.angle).toFixed(3),
        })),
      });
    }
    this._render();
  }

  // ── geometry ─────────────────────────────────────────────────────────────

  get _W() {
    return Math.max(1, (this.widthWidget && this.widthWidget.value) || 832);
  }

  get _H() {
    return Math.max(1, (this.heightWidget && this.heightWidget.value) || 1216);
  }

  /** Area fraction is canvas independent in normalised coordinates. */
  _areaOf(marker) {
    return Math.PI * marker.rx * marker.ry;
  }

  _minRx() {
    return MIN_RADIUS_PX / this._W;
  }

  _minRy() {
    return MIN_RADIUS_PX / this._H;
  }

  /** Normalised marker -> canvas pixels. */
  _toCanvas(marker) {
    return { x: marker.cx * this._W * this.scaleX, y: marker.cy * this._H * this.scaleY };
  }

  _radiiPx(marker) {
    return { rx: marker.rx * this._W * this.scaleX, ry: marker.ry * this._H * this.scaleY };
  }

  _rad(marker) {
    return ((marker.angle || 0) * Math.PI) / 180;
  }

  /** Canvas point -> the marker's local (unrotated) frame, in canvas pixels. */
  _toMarkerLocal(marker, point) {
    const p = this._toCanvas(marker);
    return toLocal(point.x - p.x, point.y - p.y, this._rad(marker));
  }

  /** The same point in canvas pixels, then in marker-local pixels. */
  _toMarkerLocalPx(marker, mouse) {
    const p = { x: marker.cx * this._W, y: marker.cy * this._H };
    return toLocal(mouse.x / this.scaleX - p.x, mouse.y / this.scaleY - p.y, this._rad(marker));
  }

  /** A circle in pixels needs rx / ry == H / W in normalised units. */
  _circleRy(rx) {
    return clamp(rx * (this._W / this._H), this._minRy(), MAX_RADIUS_NORM);
  }

  /** rx / ry for a target area fraction, keeping the marker's aspect. */
  _radiiForArea(area, aspect) {
    // pi * rx * ry = area, ry = aspect * rx  ->  rx = sqrt(area / (pi * aspect))
    const r = Math.sqrt(area / (Math.PI * Math.max(aspect, 1e-6)));
    return {
      rx: clamp(r, this._minRx(), MAX_RADIUS_NORM),
      ry: clamp(r * aspect, this._minRy(), MAX_RADIUS_NORM),
    };
  }

  _newMarkerAt(cx, cy) {
    // Circle on screen.  A new marker takes an equal share of the "medium"
    // class area, so a pair lands on the class median rather than doubling it.
    const share = PRESETS[2].area / (this.markers.length + 1);
    const rx = Math.sqrt(share / (Math.PI * (this._W / this._H)));
    return { cx: clamp(cx, 0, 1), cy: clamp(cy, 0, 1), rx, ry: this._circleRy(rx), angle: 0 };
  }

  _addMarkerAt(cx, cy) {
    if (this.markers.length >= MAX_MARKERS) return;
    if (cx === undefined) {
      // No click position: place it to the right of the last marker, or centred.
      const last = this.markers[this.markers.length - 1];
      cx = last ? clamp(last.cx + 0.29, 0.05, 0.95) : 0.355;
      cy = last ? last.cy : 0.46;
    }
    this.markers.push(this._newMarkerAt(cx, cy));
    this.selected = this.markers.length - 1;
    // Keep the drawing on the medium class: every marker takes an equal share
    // of it, so adding a second marker shrinks the first instead of leaving the
    // pair at one and a half times the class area.
    const share = PRESETS[2].area / this.markers.length;
    for (const marker of this.markers) {
      const aspect = marker.ry / Math.max(marker.rx, 1e-9);
      const { rx, ry } = this._radiiForArea(share, aspect);
      marker.rx = rx;
      marker.ry = ry;
    }
    this._commit();
  }

  /** Area fraction of the whole drawing -- what the size classes are measured on. */
  _totalArea() {
    return this.markers.reduce((sum, marker) => sum + this._areaOf(marker), 0);
  }

  /**
   * The class medians in analysis.md are TOTAL marker area fractions, so a
   * preset rescales the whole drawing to the class area and keeps the markers'
   * relative sizes.  Choosing a class must never double the area just because
   * two markers happen to be on the canvas.
   */
  _applyPreset(preset) {
    if (!this.markers.length) return;
    const areas = this.markers.map((marker) => this._areaOf(marker));
    const total = areas.reduce((a, b) => a + b, 0);
    const shares = total > 0
      ? areas.map((area) => area / total)
      : this.markers.map(() => 1 / this.markers.length);
    this.markers.forEach((marker, index) => {
      const aspect = marker.ry / Math.max(marker.rx, 1e-9);
      const { rx, ry } = this._radiiForArea(preset.area * shares[index], aspect);
      marker.rx = rx;
      marker.ry = ry;
    });
    this._commit();
  }

  // ── layout ───────────────────────────────────────────────────────────────

  _measureControls(reflow) {
    const measured = this.controls ? this.controls.offsetHeight : 0;
    if (measured > 0) {
      const next = Math.ceil(measured) + 4;
      if (next !== this.node._bscControlsH) {
        this.node._bscControlsH = next;
        if (reflow) this._resizeNode();
      }
    }
  }

  _relayout() {
    const avail = Math.max(140, Math.round((this.node.size ? this.node.size[0] : 420) - 45));
    let displayW = Math.min(avail, MAX_DISPLAY);
    let displayH = Math.max(1, Math.round(displayW * (this._H / this._W)));
    if (displayH > MAX_DISPLAY) {
      const k = MAX_DISPLAY / displayH;
      displayH = MAX_DISPLAY;
      displayW = Math.max(1, Math.round(displayW * k));
    }
    if (this.displayW === displayW && this.displayH === displayH) return;

    this.displayW = displayW;
    this.displayH = displayH;
    this.scaleX = displayW / this._W;
    this.scaleY = displayH / this._H;

    this.canvas.width = displayW + MARGIN * 2;
    this.canvas.height = displayH + MARGIN * 2;
    this.canvas.style.width = `${this.canvas.width}px`;
    this.canvas.style.height = `${this.canvas.height}px`;
    this.node._bscCanvasH = this.canvas.height;

    this._resizeNode();
    this._render();
  }

  _resizeNode() {
    const total = this.node._bscCanvasH + this.node._bscControlsH + this._chromeHeight();
    const width = Math.max(this.node.size[0], 300);
    this.node.setSize([width, total]);
    if (this.node.graph) this.node.graph.setDirtyCanvas(true, true);
  }

  /** Height taken by the title bar plus the node's own value widgets. */
  _chromeHeight() {
    let rows = 0;
    for (const widget of this.node.widgets || []) {
      if (widget === this.node._bscCanvasWidget || widget === this.node._bscControlsWidget) continue;
      if (widget.hidden) continue;
      rows += 1;
    }
    // Over-estimating is safe (a little slack under the canvas); under-estimating
    // would push the canvas out of the node body.
    return 34 + rows * 44 + 16;
  }

  // ── events ───────────────────────────────────────────────────────────────

  _setupEvents() {
    this._onPointerDown = (e) => this.onPointerDown(e);
    this._onPointerMove = (e) => this.onPointerMove(e);
    this._onPointerUp = (e) => this.onPointerUp(e);
    this._onWheel = (e) => this.onWheel(e);
    this._onContextMenu = (e) => e.preventDefault();
    this._onPointerLeave = () => {
      if (!this.drag) this.canvas.style.cursor = "default";
    };

    this.canvas.addEventListener("pointerdown", this._onPointerDown);
    this.canvas.addEventListener("pointermove", this._onPointerMove);
    this.canvas.addEventListener("pointerup", this._onPointerUp);
    this.canvas.addEventListener("pointercancel", this._onPointerUp);
    this.canvas.addEventListener("wheel", this._onWheel, { passive: false });
    this.canvas.addEventListener("contextmenu", this._onContextMenu);
    this.canvas.addEventListener("pointerleave", this._onPointerLeave);
  }

  destroy() {
    this.canvas.removeEventListener("pointerdown", this._onPointerDown);
    this.canvas.removeEventListener("pointermove", this._onPointerMove);
    this.canvas.removeEventListener("pointerup", this._onPointerUp);
    this.canvas.removeEventListener("pointercancel", this._onPointerUp);
    this.canvas.removeEventListener("wheel", this._onWheel);
    this.canvas.removeEventListener("contextmenu", this._onContextMenu);
    this.canvas.removeEventListener("pointerleave", this._onPointerLeave);
    this._unwatchSourceWidget();
  }

  /** Mouse position in canvas backing-store pixels, origin at the drawing area. */
  _mouse(e) {
    const rect = this.canvas.getBoundingClientRect();
    const sx = rect.width ? this.canvas.width / rect.width : 1;
    const sy = rect.height ? this.canvas.height / rect.height : 1;
    return {
      x: (e.clientX - rect.left) * sx - MARGIN,
      y: (e.clientY - rect.top) * sy - MARGIN,
      rawX: (e.clientX - rect.left) * sx,
      rawY: (e.clientY - rect.top) * sy,
    };
  }

  /**
   * Handle positions in canvas pixels, following the marker's rotation:
   * local (lx, ly) maps to centre + (lx*cos - ly*sin, lx*sin + ly*cos), so the
   * edge handles always sit on the ellipse's own axes.
   */
  _handles(marker) {
    const p = this._toCanvas(marker);
    const { rx, ry } = this._radiiPx(marker);
    const radians = this._rad(marker);
    const c = Math.cos(radians);
    const s = Math.sin(radians);
    const at = (lx, ly) => ({ x: p.x + lx * c - ly * s, y: p.y + lx * s + ly * c });
    const armLength = ry + ROTATE_ARM;
    return [
      { id: "e", ...at(rx, 0), mode: "rx" },
      { id: "w", ...at(-rx, 0), mode: "rx" },
      { id: "n", ...at(0, -ry), mode: "ry" },
      { id: "s", ...at(0, ry), mode: "ry" },
      { id: "ne", ...at(rx, -ry), mode: "both" },
      { id: "nw", ...at(-rx, -ry), mode: "both" },
      { id: "se", ...at(rx, ry), mode: "both" },
      { id: "sw", ...at(-rx, ry), mode: "both" },
      { id: "rot", ...at(0, -armLength), mode: "rotate" },
    ];
  }

  _hitHandle(mouse) {
    if (this.selected < 0) return null;
    const marker = this.markers[this.selected];
    if (!marker) return null;
    // Handles are drawn (and hit) for the selected marker only.  The rotation
    // handle gets a slightly larger target than the square resize handles.
    for (const handle of this._handles(marker)) {
      const tolerance = handle.id === "rot" ? 9 : 7;
      if (Math.hypot(handle.x - mouse.x, handle.y - mouse.y) <= tolerance) {
        return { index: this.selected, handle };
      }
    }
    return null;
  }

  _hitBody(mouse) {
    for (let i = this.markers.length - 1; i >= 0; i--) {
      const marker = this.markers[i];
      const p = this._toCanvas(marker);
      const { rx, ry } = this._radiiPx(marker);
      const local = toLocal(mouse.x - p.x, mouse.y - p.y, this._rad(marker));
      const dx = local.x / Math.max(rx, 0.5);
      const dy = local.y / Math.max(ry, 0.5);
      if (dx * dx + dy * dy <= 1.0) return i;
    }
    return -1;
  }

  onPointerDown(e) {
    e.preventDefault();
    e.stopPropagation();
    if (e.button === 2) {
      const body = this._hitBody(this._mouse(e));
      if (body >= 0) {
        this.markers.splice(body, 1);
        this.selected = this.markers.length ? Math.min(body, this.markers.length - 1) : -1;
        this._commit();
      }
      return;
    }
    if (e.button !== 0) return;

    const mouse = this._mouse(e);
    const inside = mouse.x >= 0 && mouse.y >= 0 && mouse.x <= this.displayW && mouse.y <= this.displayH;
    if (!inside) return;

    const handleHit = this._hitHandle(mouse);
    if (handleHit) {
      const marker = this.markers[handleHit.index];
      this.drag = {
        index: handleHit.index,
        handle: handleHit.handle,
        mode: handleHit.handle.mode,
        uniform: handleHit.handle.mode === "both" || e.shiftKey,
        start: { ...marker },
        mouse,
      };
      this.canvas.setPointerCapture?.(e.pointerId);
      this.canvas.style.cursor = this._cursorFor(handleHit.handle.id);
      return;
    }

    const body = this._hitBody(mouse);
    if (body >= 0) {
      this.selected = body;
      const marker = this.markers[body];
      const p = this._toCanvas(marker);
      this.drag = {
        index: body,
        mode: "move",
        offset: { x: p.x - mouse.x, y: p.y - mouse.y },
      };
      this.canvas.setPointerCapture?.(e.pointerId);
      this.canvas.style.cursor = "move";
      this._render();
      return;
    }

    // Empty space: create a marker there (up to the two-marker contract).
    if (this.markers.length < MAX_MARKERS) {
      const cx = clamp(mouse.x / this.scaleX / this._W, 0, 1);
      const cy = clamp(mouse.y / this.scaleY / this._H, 0, 1);
      this._addMarkerAt(cx, cy);
      const marker = this.markers[this.markers.length - 1];
      const p = this._toCanvas(marker);
      this.drag = {
        index: this.markers.length - 1,
        mode: "move",
        offset: { x: p.x - mouse.x, y: p.y - mouse.y },
      };
      this.canvas.setPointerCapture?.(e.pointerId);
      this.canvas.style.cursor = "move";
    }
  }

  onPointerMove(e) {
    const mouse = this._mouse(e);
    if (!this.drag) {
      const handleHit = this._hitHandle(mouse);
      if (handleHit) {
        this.canvas.style.cursor = this._cursorFor(handleHit.handle.id);
      } else if (this._hitBody(mouse) >= 0) {
        this.canvas.style.cursor = "move";
      } else {
        this.canvas.style.cursor = this.markers.length < MAX_MARKERS ? "crosshair" : "default";
      }
      return;
    }

    const marker = this.markers[this.drag.index];
    if (!marker) return;
    e.stopPropagation();

    if (this.drag.mode === "move") {
      const x = mouse.x + this.drag.offset.x;
      const y = mouse.y + this.drag.offset.y;
      marker.cx = clamp(x / this.scaleX / this._W, 0, 1);
      marker.cy = clamp(y / this.scaleY / this._H, 0, 1);
      this._commit();
      return;
    }

    const start = this.drag.start;
    const startRadians = (start.angle * Math.PI) / 180;
    const centre = {
      x: start.cx * this._W * this.scaleX,
      y: start.cy * this._H * this.scaleY,
    };

    if (this.drag.mode === "rotate") {
      // The rotation handle sits on the marker's local "up" axis, so the angle
      // is the direction of the pointer relative to that axis.  Measured in
      // canvas space, which is what the user actually aims at.
      const dx = mouse.x - centre.x;
      const dy = mouse.y - centre.y;
      if (Math.hypot(dx, dy) < 2) return;
      marker.angle = normalizeAngle((Math.atan2(dx, -dy) * 180) / Math.PI);
      this._commit();
      return;
    }

    if (this.drag.uniform) {
      const startDist = Math.hypot(
        this.drag.mouse.x - centre.x,
        this.drag.mouse.y - centre.y,
      );
      const dist = Math.hypot(mouse.x - centre.x, mouse.y - centre.y);
      const factor = startDist > 1 ? dist / startDist : 1;
      marker.rx = clamp(start.rx * factor, this._minRx(), MAX_RADIUS_NORM);
      marker.ry = clamp(start.ry * factor, this._minRy(), MAX_RADIUS_NORM);
      this._commit();
      return;
    }

    // Edge handles resize along the marker's own (rotated) axes, so the maths
    // happens in the local frame, in pixels.
    const local = this._toMarkerLocalPx(start, mouse);
    if (this.drag.mode === "rx") {
      marker.rx = clamp(Math.abs(local.x) / this._W, this._minRx(), MAX_RADIUS_NORM);
    } else if (this.drag.mode === "ry") {
      marker.ry = clamp(Math.abs(local.y) / this._H, this._minRy(), MAX_RADIUS_NORM);
    }
    this._commit();
  }

  onPointerUp(e) {
    if (!this.drag) return;
    this.drag = null;
    this.canvas.releasePointerCapture?.(e.pointerId);
    this.canvas.style.cursor = "default";
    this._measureControls(true);
    this._render();
  }

  onWheel(e) {
    const mouse = this._mouse(e);
    const body = this._hitBody(mouse);
    if (body < 0) return; // let the graph zoom when the wheel is not over a marker
    e.preventDefault();
    e.stopPropagation();
    this.selected = body;
    const marker = this.markers[body];
    const factor = e.deltaY < 0 ? 1.06 : 1 / 1.06;
    marker.rx = clamp(marker.rx * factor, this._minRx(), MAX_RADIUS_NORM);
    marker.ry = clamp(marker.ry * factor, this._minRy(), MAX_RADIUS_NORM);
    this._commit();
  }

  _cursorFor(handleId) {
    switch (handleId) {
      case "n":
      case "s":
        return "ns-resize";
      case "e":
      case "w":
        return "ew-resize";
      case "ne":
      case "sw":
        return "nesw-resize";
      case "rot":
        return "grab";
      default:
        return "nwse-resize";
    }
  }

  // ── rendering ────────────────────────────────────────────────────────────

  _render() {
    const ctx = this.ctx;
    if (!ctx) return;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    // Area outside the canvas: the node's own backdrop.
    ctx.fillStyle = "#1b1b1b";
    ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

    ctx.save();
    ctx.translate(MARGIN, MARGIN);

    // The real output pixels start here: black background, coloured markers.
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, this.displayW, this.displayH);

    if (this.guideImg && this.guideEnabled) {
      ctx.globalAlpha = 0.45;
      try {
        ctx.drawImage(this.guideImg, 0, 0, this.displayW, this.displayH);
      } catch (error) {
        /* an unloaded guide is not fatal */
      }
      ctx.globalAlpha = 1;
    }

    for (let i = 0; i < this.markers.length; i++) {
      const marker = this.markers[i];
      const p = this._toCanvas(marker);
      const { rx, ry } = this._radiiPx(marker);
      const radians = this._rad(marker);
      ctx.beginPath();
      ctx.ellipse(p.x, p.y, Math.max(rx, 0.01), Math.max(ry, 0.01), radians, 0, Math.PI * 2);
      ctx.fillStyle = this.color;
      ctx.fill();

      const isSelected = i === this.selected;
      ctx.setLineDash(isSelected ? [6, 4] : [3, 3]);
      ctx.strokeStyle = isSelected ? "rgba(255,255,255,0.95)" : "rgba(140,180,255,0.55)";
      ctx.lineWidth = isSelected ? 1.5 : 1;
      ctx.stroke();
      ctx.setLineDash([]);

      if (isSelected) {
        const handles = this._handles(marker);
        // Stem from the top of the ellipse out to the rotation handle.
        const rot = handles[handles.length - 1];
        const north = handles.find((handle) => handle.id === "n");
        if (north && rot) {
          ctx.beginPath();
          ctx.moveTo(north.x, north.y);
          ctx.lineTo(rot.x, rot.y);
          ctx.strokeStyle = "rgba(255,255,255,0.7)";
          ctx.lineWidth = 1;
          ctx.stroke();
        }
        for (const handle of handles) {
          if (handle.id === "rot") {
            ctx.beginPath();
            ctx.arc(handle.x, handle.y, 5.5, 0, Math.PI * 2);
            ctx.fillStyle = "#7fd4ff";
            ctx.fill();
            ctx.strokeStyle = "#101010";
            ctx.lineWidth = 1;
            ctx.stroke();
          } else {
            ctx.beginPath();
            ctx.rect(handle.x - 3.5, handle.y - 3.5, 7, 7);
            ctx.fillStyle = "#ffffff";
            ctx.fill();
            ctx.strokeStyle = "#101010";
            ctx.lineWidth = 1;
            ctx.stroke();
          }
        }
      }
    }

    // Canvas border, drawn last so it is never covered by a marker.
    ctx.strokeStyle = "rgba(120,120,120,0.85)";
    ctx.lineWidth = 1;
    ctx.strokeRect(0.5, 0.5, this.displayW - 1, this.displayH - 1);

    ctx.restore();
    this._updateStatus();
  }

  _updateStatus() {
    if (!this.status) return;
    this._updateAngleReadout();
    const count = this.markers.length;
    if (!count) {
      this.status.textContent = "no markers - click the canvas to add one";
      return;
    }
    const total = this._totalArea();
    const nearest = PRESETS.reduce(
      (best, preset) =>
        Math.abs(Math.log(preset.area / total)) < Math.abs(Math.log(best.area / total)) ? preset : best,
      PRESETS[0],
    );
    const index = this.selected >= 0 && this.markers[this.selected] ? this.selected : -1;
    let perMarker = "";
    if (index >= 0) {
      const marker = this.markers[index];
      const angle = normalizeAngle(marker.angle || 0);
      perMarker = ` - marker ${index + 1}/${count}: ${(this._areaOf(marker) * 100).toFixed(2)}%` +
        (angle ? ` @ ${angle.toFixed(1)}°` : "");
    }
    this.status.textContent =
      `${this._W}x${this._H} - total ${(total * 100).toFixed(2)}% ` +
      `(${nearest.name})${perMarker}`;
  }

  // ── size integration ─────────────────────────────────────────────────────

  /** Called with the node's `ui` payload after a run: the authoritative size. */
  applyExecutedSize(width, height) {
    if (Array.isArray(width)) width = width[0];
    if (Array.isArray(height)) height = height[0];
    if (!width || !height) return;
    this._setSize(Math.round(width), Math.round(height));
  }

  /** Re-read the widgets after the workflow restored them (onConfigure). */
  refresh() {
    this._readWidgets();
    this._readState();
    this.displayW = undefined; // the restored size may differ from the created one
    this._relayout();
    this._commit();
    this._syncFromConnectedImage();
  }

  _setSize(width, height) {
    if (!(width > 0 && height > 0)) return;
    if (width === this._W && height === this._H) return;
    const oldW = this._W;
    const oldH = this._H;
    if (this.widthWidget) this.widthWidget.value = width;
    if (this.heightWidget) this.heightWidget.value = height;
    // Radii are normalised per axis, so keep the pixel aspect when the canvas
    // aspect changes: a circle stays a circle instead of shearing into an
    // ellipse just because the canvas got wider or taller.
    const factor = (oldH * width) / (oldW * height);
    if (Number.isFinite(factor) && factor > 0) {
      for (const marker of this.markers) {
        marker.ry = clamp(marker.ry * factor, this._minRy(), MAX_RADIUS_NORM);
      }
    }
    this.displayW = undefined; // force a re-layout
    this._relayout();
    this._commit();
  }

  /**
   * Best effort: read the size of a connected LoadImage-family source and use
   * it as the canvas size, and keep the image as a dimmed underlay, so the
   * drawing is already correct before the first run.  The node's own `ui`
   * output stays authoritative.
   *
   * Mirrors ComfyUI-KJNodes' `resolveSourcePreview`: on current frontends
   * `graph.links` is a Map, the file widget value may carry a subfolder, and a
   * node that already ran exposes its preview through `node.imgs`.
   */
  _syncFromConnectedImage() {
    try {
      if (!this.node.graph || !this.node.inputs) return;
      const input = this.node.inputs.find((i) => i.name === "image");
      if (!input || input.link === null || input.link === undefined) return;

      const links = this.node.graph.links;
      const link = links && typeof links.get === "function" ? links.get(input.link) : links?.[input.link];
      if (!link) return;
      const origin = this.node.graph.getNodeById(link.origin_id);
      if (!origin) return;

      const fileWidget = (origin.widgets || []).find((w) => w.name === "image" || w.name === "video");
      let url = null;
      if (fileWidget && typeof fileWidget.value === "string" && fileWidget.value) {
        this._watchSourceWidget(fileWidget);
        const cleaned = fileWidget.value.replace(/\s*\[(input|output|temp)\]\s*$/i, "");
        const slash = cleaned.lastIndexOf("/");
        const subfolder = slash >= 0 ? cleaned.slice(0, slash) : "";
        const filename = slash >= 0 ? cleaned.slice(slash + 1) : cleaned;
        url = `/view?filename=${encodeURIComponent(filename)}&type=input` +
              `&subfolder=${encodeURIComponent(subfolder)}&no-cache=${Date.now()}`;
      } else if (origin.imgs && origin.imgs.length > 0 && origin.imgs[0].src) {
        url = origin.imgs[0].src; // available once the source node has run
      }
      if (!url) return;

      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => {
        if (!img.naturalWidth || !img.naturalHeight) return;
        this.guideImg = img;
        this._setSize(img.naturalWidth, img.naturalHeight);
        this._render();
      };
      img.onerror = () => {
        /* not a viewable file widget - the executed size will fix it */
      };
      img.src = url;
    } catch (error) {
      console.warn("[BSC] could not read the connected image size", error);
    }
  }

  /** Re-sync when the user picks a different file in the source node. */
  _watchSourceWidget(widget) {
    if (this._watchedWidget === widget) return;
    this._unwatchSourceWidget();
    const original = widget.callback;
    widget.callback = (...args) => {
      const result = typeof original === "function" ? original.apply(widget, args) : undefined;
      setTimeout(() => this._syncFromConnectedImage(), 0);
      return result;
    };
    this._watchedWidget = widget;
    this._watchedCallback = original;
  }

  _unwatchSourceWidget() {
    if (this._watchedWidget) {
      this._watchedWidget.callback = this._watchedCallback;
      this._watchedWidget = null;
      this._watchedCallback = undefined;
    }
  }
}

// ── registration ───────────────────────────────────────────────────────────

async function getApp() {
  // Modern frontends expose the app object here; this is the primary path.
  if (window.comfyAPI && window.comfyAPI.app && window.comfyAPI.app.app) {
    return window.comfyAPI.app.app;
  }
  // Fallback for older frontends.  This file is served from
  // /extensions/<node dir>/js/, so the module-relative depth is not obvious --
  // try the absolute path first (app.js always lives at /scripts/app.js) and
  // then both plausible relative depths.
  const candidates = ["/scripts/app.js", "../../../scripts/app.js", "../../scripts/app.js"];
  for (const specifier of candidates) {
    try {
      const module = await import(/* @vite-ignore */ specifier);
      if (module && module.app) return module.app;
    } catch (error) {
      /* try the next candidate */
    }
  }
  throw new Error("could not locate ComfyUI's /scripts/app.js");
}

getApp()
  .then((app) => {
    app.registerExtension({
      name: "EditUtils.BreastSizeControl",
      async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!nodeData || nodeData.name !== NODE_NAME) return;

        chainCallback(nodeType.prototype, "onNodeCreated", function () {
          if (this._bscEditor) this._bscEditor.destroy();
          this._bscEditor = new BreastSizeEditor(this);
        });

        chainCallback(nodeType.prototype, "onConfigure", function () {
          const editor = this._bscEditor;
          if (!editor) return;
          // Widget values are restored before onConfigure runs, so the drawing
          // and the canvas size have to be taken from them again.
          setTimeout(() => editor.refresh(), 0);
        });

        chainCallback(nodeType.prototype, "onResize", function () {
          if (this._bscEditor) this._bscEditor._relayout();
        });

        chainCallback(nodeType.prototype, "onConnectionsChange", function () {
          if (this._bscEditor) setTimeout(() => this._bscEditor._syncFromConnectedImage(), 50);
        });

        chainCallback(nodeType.prototype, "onExecuted", function (message) {
          if (this._bscEditor && message) {
            this._bscEditor.applyExecutedSize(message.width, message.height);
          }
        });

        chainCallback(nodeType.prototype, "onRemoved", function () {
          if (this._bscEditor) {
            this._bscEditor.destroy();
            this._bscEditor = null;
          }
        });
      },
    });
  })
  .catch((error) => console.error("[BSC] failed to register the canvas editor", error));
