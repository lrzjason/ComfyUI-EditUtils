/**
 * ComfyUI-EditUtils -- Flux3KleinBBox_EditUtils bounding-box editor.
 *
 * FLUX.3 takes its layout as an "element table": a JSON array appended to the
 * prompt, one row per element. This canvas is how those rows are drawn: you
 * draw a box on the reference, and the row's id and description are typed next
 * to it. The coordinates follow the documented FLUX.3 contract exactly --
 * [top, left, bottom, right], integers 0..1000 from the top-left corner -- so a
 * box is independent of the image's size and aspect ratio.
 *
 * This node is deliberately SEPARATE from Flux3KleinConfigPreparer_EditUtils:
 * the preparer stays a short, ordinary config node, and all the drawing (and the
 * tall row-editing area) lives here. The node takes the config chain plus one
 * image's config, attaches the table to that config, and returns the chain.
 *
 * Two row schemas, matching the FLUX.3 docs:
 *   generation  {"id","bbox","desc"}
 *               "Draw a box, describe what belongs in it."
 *   editing     {"id","from","src_bbox","tgt_bbox","desc"}
 *
 * NAMING, because the editing row reads as if it repeats itself:
 * `from`, `ref` and `src` are all the SAME thing - the input image an element
 * comes from. `from` is the field name, `ref_image_0` is its value (the image),
 * and `src_bbox` is named after that image: "the box in the source". So an
 * editing row carries only three real facts:
 *   from       which image it comes from   (null = it is not from an image)
 *   src_bbox   where it is in that image   (the box you see now)
 *   tgt_bbox   where it goes in the result (null = delete it)
 *
 * The four row kinds are those three facts in combination:
 *   Keep     from=image, src_bbox == tgt_bbox   leave it exactly where it is
 *   Move     from=image, src_bbox != tgt_bbox   move or resize it
 *   New      from=null,  src_bbox=null, tgt_bbox set   create it (add/replace/recolour)
 *   Remove   from=image, tgt_bbox=null          delete it
 *
 * In editing mode each row therefore has TWO boxes: the source box (where the
 * element is now, drawn solid over the reference) and the target box (where it
 * should end up, drawn dashed). Dragging on empty canvas draws whichever box
 * the selected row/field is currently editing.
 *
 * The payload lives in the hidden `bbox_rows` widget so a workflow reload
 * restores ids, descriptions and both boxes exactly.
 *
 * Geometry and layout follow ComfyUI-KJNodes' canvas editors (the same pattern
 * web/js/breast_size_control.js uses); this file is self-contained.
 */

const NODE_NAME = "Flux3KleinBBox_EditUtils";

const MARGIN = 14;
const MAX_DISPLAY = 1024;
const HANDLE = 5;             // corner/edge handle half-size in canvas px
const MIN_BOX_PX = 4;         // a box never gets smaller than this on screen
const GRID = 1000;            // FLUX.3 boxes are 0..1000 on both axes
// The row table is the main editing surface, so it gets a tall scrollable
// area (this node exists precisely to give the rows room to breathe).
const ROWS_MAX_H = 420;

// Palette: one colour per row (cycled). Kept distinguishable on both light and
// dark references, with the label drawn on a dark plate for legibility.
const ROW_COLORS = [
  "#ff5a5f", "#2ec4b6", "#ffb703", "#8e7dff", "#4cc9f0",
  "#f72585", "#90be6d", "#ff8fab", "#00b4d8", "#e0aaff",
];
const SRC_COLOR = "#ffd166";  // editing: the source box (where it is now)
const TGT_COLOR = "#06d6a0";  // editing: the target box (where it goes)

// The four kinds of edit row (FLUX.3's own words, from its field table).
// `hint` is the plain-language meaning, shown in the legend and on the button,
// so the buttons do not require reading the spec's field shorthand.
const KINDS = [
  {
    name: "Keep",
    hint: "leave it exactly where it is",
    long: "Keep: the element already exists in the source image and stays in the same place. " +
          "Both boxes hold the same coordinates.",
  },
  {
    name: "Move",
    hint: "move or resize it",
    long: "Move: the element already exists in the source image but gets a different target box, " +
          "so it moves or changes size.",
  },
  {
    name: "New",
    hint: "create it (add / replace / recolour)",
    long: "New: the element is NOT taken from the source image - it is created inside the target " +
          "box from its description. Use it to add, replace or recolour something.",
  },
  {
    name: "Remove",
    hint: "delete it",
    long: "Remove: the element exists in the source image but has no target box, so the model " +
          "deletes it and fills the area with whatever was behind it.",
  },
];

// The element-table schemas. No `disabled`: the node is dedicated to the table,
// so its default is the simplest useful mode.
const MODES = ["generation", "editing"];
const DEFAULT_MODE = "generation";

// Every element field is a full-width labelled line, so a long id, reference or
// description is readable without horizontal scrolling.
//   id / from ref / item desc  ->  one line each
const FIELD_LABEL_PX = 64;
const INPUT_CSS =
  "width:100%;box-sizing:border-box;height:22px;font:11px sans-serif;" +
  "background:#1e1e1e;color:#ddd;border:1px solid #555;border-radius:3px;padding:0 5px;";
const TEXTAREA_CSS =
  "width:100%;box-sizing:border-box;font:11px sans-serif;background:#1e1e1e;color:#ddd;" +
  "border:1px solid #555;border-radius:3px;padding:3px 5px;resize:vertical;min-height:38px;";

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

/** FLUX.3 ints: clamp to 0..1000 and round. NaN stays NaN (rejected upstream). */
function toGrid(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return NaN;
  return Math.round(clamp(n, 0, GRID));
}

/**
 * A box as {t,l,b,r} in grid units, normalized so t<=b and l<=r.
 * Returns null when any coordinate is missing or non-finite, so an unusable box
 * is rejected instead of becoming a zero box that could compare equal to another
 * unusable box.
 */
function normalizeBox(box) {
  if (!box || typeof box !== "object") return null;
  const t = toGrid(box.t), l = toGrid(box.l), b = toGrid(box.b), r = toGrid(box.r);
  if (![t, l, b, r].every(Number.isFinite)) return null;
  return { t: Math.min(t, b), b: Math.max(t, b), l: Math.min(l, r), r: Math.max(l, r) };
}

/**
 * Accept a box in any of the shapes that reach this editor: the internal
 * {t,l,b,r} object, or the documented [top,left,bottom,right] array. Returns a
 * normalized {t,l,b,r} object, or null when the value is not a usable box.
 * Without this, a raw array would read as undefined fields and silently turn
 * into a zero box, which would misclassify rows.
 */
function asBox(value) {
  if (!value) return null;
  if (Array.isArray(value)) return arrayToBox(value);
  if (typeof value === "object") return normalizeBox(value);
  return null;
}

/**
 * Grid box -> the documented [top, left, bottom, right] array.
 * Returns null for anything that is not a usable box (never a phantom
 * [null,null,null,null], which would compare equal to another such phantom).
 */
function boxToArray(box) {
  const n = asBox(box);
  if (!n) return null;
  return [n.t, n.l, n.b, n.r];
}

function arrayToBox(arr) {
  if (!Array.isArray(arr) || arr.length !== 4) return null;
  const nums = arr.map(Number);
  if (!nums.every(Number.isFinite)) return null;
  return normalizeBox({ t: nums[0], l: nums[1], b: nums[2], r: nums[3] });
}

function boxArea(box) {
  const n = asBox(box);
  if (!n) return 0;
  return (n.b - n.t) * (n.r - n.l);
}

/**
 * Row kind from its fields -- the same classification the Python node applies.
 * Uses a value comparison on the coordinate arrays; two unusable boxes are NOT
 * treated as equal, so a malformed row reads as Move rather than Keep.
 */
function rowKind(row) {
  if (!row.from) return "New";
  const s = boxToArray(row.src_bbox);
  if (!row.tgt_bbox) return "Remove";
  const t = boxToArray(row.tgt_bbox);
  if (!s || !t) return "Move";   // unusable boxes must not compare equal
  return s.join() === t.join() ? "Keep" : "Move";
}

function max0(value) {
  return Math.max(0, value);
}

let idCounter = 0;
function suggestId(prefix) {
  idCounter += 1;
  return `${prefix || "element"}_${idCounter}`;
}

class Flux3BboxEditor {
  constructor(node) {
    this.node = node;
    this.canvas = document.createElement("canvas");
    this.ctx = this.canvas.getContext("2d");

    this.rows = [];
    this.selected = 0;
    this.mode = DEFAULT_MODE;
    this.field = "tgt";      // editing: which box the canvas drag edits
    this.guideImg = null;
    this.guideEnabled = true;
    this.showLabels = true;
    this.drag = null;

    this.node._f3CanvasH = 420;
    this.node._f3ControlsH = ROWS_MAX_H + 120;   // rows + action/field rows
    this.node._f3RowsH = ROWS_MAX_H;

    this._readWidgets();
    this._buildDom();
    this._readState();
    this._relayout();
    this._commit();

    // ComfyUI settles the node geometry and the DOM-widget boxes over the next
    // few frames, so the first layout above is often computed from a stale node
    // width (the canvas then looks blank until something else redraws it). Run a
    // few cheap layout passes across the next ~600 ms; `_resizeNode` skips
    // no-op resizes, so repeating these is harmless.
    requestAnimationFrame(() => this.refreshLayout());
    for (const delay of [60, 200, 600]) {
      setTimeout(() => {
        if (!this.destroyed) this.refreshLayout();
      }, delay);
    }
  }

  // ── widgets & DOM ────────────────────────────────────────────────────────

  _readWidgets() {
    const find = (name) => (this.node.widgets || []).find((w) => w.name === name);
    this.modeWidget = find("bbox_mode");
    this.rowsWidget = find("bbox_rows");
    this.idsWidget = find("bbox_ids");
    this.descsWidget = find("bbox_descs");
    // The row payload is the drawing itself: edited on the canvas, never by hand.
    for (const widget of [this.rowsWidget, this.idsWidget, this.descsWidget]) {
      if (!widget) continue;
      widget.hidden = true;
      widget.options = widget.options || {};
      widget.options.hidden = true;
    }
    // `bbox_mode` deliberately stays VISIBLE: it is a real setting, so hiding it
    // would leave no way to change it if this script ever failed to load. The
    // canvas just follows it instead of drawing its own duplicate control.
    this._watchModeWidget();
  }

  /**
   * Follow the native `bbox_mode` widget so switching it on the node updates the
   * canvas immediately (and clears the row-kind semantics that only apply to
   * `editing`).
   */
  _watchModeWidget() {
    const widget = this.modeWidget;
    if (!widget || this._watchedModeWidget === widget) return;
    const original = widget.callback;
    this._watchedModeWidget = widget;
    this._watchedModeCallback = original;
    widget.callback = (...args) => {
      const result = typeof original === "function" ? original.apply(widget, args) : undefined;
      const next = MODES.includes(widget.value) ? widget.value : DEFAULT_MODE;
      if (next !== this.mode) {
        this.mode = next;
        this._commit();
      }
      return result;
    };
  }

  _buildDom() {
    const canvasWrap = document.createElement("div");
    canvasWrap.style.cssText =
      "position:relative;display:flex;justify-content:center;align-items:center;" +
      "width:100%;box-sizing:border-box;overflow:hidden;";
    canvasWrap.appendChild(this.canvas);
    this.canvas.style.cssText = "display:block;touch-action:none;cursor:crosshair;";

    this.controls = document.createElement("div");
    this.controls.style.cssText =
      "display:flex;flex-wrap:wrap;gap:4px;align-items:center;width:100%;" +
      "box-sizing:border-box;padding:2px 0;font:11px sans-serif;color:#bbb;";

    this._addRow(this._buildActionRow());
    this._addRow(this._buildFieldRow());

    // Row table: id (single-line) + desc (multi-line), one row per element.
    this.table = document.createElement("div");
    this.table.style.cssText =
      "display:flex;flex-direction:column;gap:3px;width:100%;box-sizing:border-box;" +
      `max-height:${ROWS_MAX_H}px;overflow-y:auto;padding-right:2px;`;
    this.controls.appendChild(this.table);

    this.status = document.createElement("span");
    this.status.style.cssText = "width:100%;font:11px sans-serif;color:#8ab4f8;";
    this.controls.appendChild(this.status);

    this.node._f3CanvasWidget = this.node.addDOMWidget(
      "f3_canvas", "F3BboxCanvas", canvasWrap,
      {
        serialize: false, hideOnZoom: false,
        getMinHeight: () => this.node._f3CanvasH,
        getMaxHeight: () => this.node._f3CanvasH,
        getHeight: () => this.node._f3CanvasH,
      },
    );
    this.node._f3ControlsWidget = this.node.addDOMWidget(
      "f3_controls", "F3BboxControls", this.controls,
      {
        serialize: false, hideOnZoom: false,
        getMinHeight: () => this.node._f3ControlsH,
        getMaxHeight: () => this.node._f3ControlsH,
        getHeight: () => this.node._f3ControlsH,
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

    // No mode dropdown here: `bbox_mode` is a real ComfyUI widget and is left
    // visible on the node, so rendering a second control for it duplicated the
    // setting (and could disagree with it). The canvas reads and reacts to the
    // native widget instead - see _watchModeWidget().

    row.appendChild(this._button("+ Box", "Add an element row", () => this._addRow_(true)));
    row.appendChild(this._button("Full frame", "Add a background box covering 0,0,1000,1000",
      () => this._addBackgroundRow()));
    row.appendChild(this._button("Clear", "Remove every row", () => {
      this.rows = [];
      this.selected = 0;
      this._commit();
    }));
    row.appendChild(this._button("Copy caption", "Copy the <id> caption to the clipboard",
      () => this._copyCaption()));
    row.appendChild(this._button("Refresh",
      "Re-measure and repaint the canvas and row table. Use this if the bbox area looks blank " +
      "or stale (e.g. right after loading a workflow).",
      () => {
        this.refreshLayout();
        this._flashStatus("refreshed");
      }));

    const guideLabel = document.createElement("label");
    guideLabel.style.cssText = "display:flex;align-items:center;gap:3px;";
    guideLabel.title = "Show the connected image underneath while drawing (never part of the output)";
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
    guideLabel.appendChild(document.createTextNode("image"));
    row.appendChild(guideLabel);

    const labelsLabel = document.createElement("label");
    labelsLabel.style.cssText = "display:flex;align-items:center;gap:3px;";
    labelsLabel.title = "Draw each row's id on its box";
    this.labelsInput = document.createElement("input");
    this.labelsInput.type = "checkbox";
    this.labelsInput.checked = true;
    this.labelsInput.style.cssText = "margin:0;";
    this.labelsInput.addEventListener("pointerdown", (e) => e.stopPropagation());
    this.labelsInput.addEventListener("change", () => {
      this.showLabels = this.labelsInput.checked;
      this._render();
    });
    labelsLabel.appendChild(this.labelsInput);
    labelsLabel.appendChild(document.createTextNode("labels"));
    row.appendChild(labelsLabel);

    return row;
  }

  /** Editing mode only: which of the two boxes a canvas drag draws/moves. */
  _buildFieldRow() {
    const row = document.createElement("div");
    row.style.cssText = "display:flex;flex-wrap:wrap;gap:4px;align-items:center;width:100%;";
    this.fieldRow = row;

    const label = document.createElement("span");
    label.textContent = "drag edits";
    label.style.cssText = "color:#999;";
    row.appendChild(label);

    const mk = (value, text, color, title) => {
      const button = this._button(text, title, () => {
        this.field = value;
        this._syncFieldButtons();
        this._render();
      });
      button.dataset.field = value;
      button._baseColor = color;
      return button;
    };

    row.appendChild(mk("src", "the source box", SRC_COLOR,
      "The source box: where the element is NOW, in the source image (the solid box). " +
      "Dragging on the canvas moves or resizes this box."));
    row.appendChild(mk("tgt", "the target box", TGT_COLOR,
      "The target box: where the element should END UP in the result (the dashed box). " +
      "Dragging on the canvas moves or resizes this box."));

    // Kind selector for the selected row.
    const kindLabel = document.createElement("span");
    kindLabel.textContent = "this row means";
    kindLabel.style.cssText = "color:#999;margin-left:6px;";
    row.appendChild(kindLabel);
    for (const kind of KINDS) {
      const button = this._button(kind.name, kind.long, () => this._applyKind(kind.name));
      button.dataset.kind = kind.name;
      row.appendChild(button);
    }

    // Visible legend: the four kinds explained without needing to hover, since
    // the button tooltips alone were not enough to make them understandable.
    const legend = document.createElement("div");
    legend.style.cssText =
      "width:100%;font:10px sans-serif;color:#8a8a8a;line-height:1.5;padding:1px 0 0 2px;";
    legend.textContent = KINDS.map((k) => `${k.name} = ${k.hint}`).join("   ·   ");
    row.appendChild(legend);

    this.fieldButtons = Array.from(row.querySelectorAll("button[data-field]"));
    this.kindButtons = Array.from(row.querySelectorAll("button[data-kind]"));
    this._syncFieldButtons();
    return row;
  }

  _syncFieldButtons() {
    for (const button of this.fieldButtons || []) {
      const active = button.dataset.field === this.field;
      button.style.background = active ? "#3a5a8c" : "#2b2b2b";
      button.style.color = active ? "#fff" : "#ddd";
    }
    for (const button of this.kindButtons || []) {
      button.style.background = "#2b2b2b";
      button.style.color = "#ddd";
    }
    const row = this.rows[this.selected];
    if (row && this.kindButtons) {
      const kind = rowKind(row);
      for (const button of this.kindButtons) {
        if (button.dataset.kind === kind) {
          button.style.background = "#3a5a8c";
          button.style.color = "#fff";
        }
      }
    }
  }

  // ── state <-> widgets ────────────────────────────────────────────────────

  _readState() {
    // A workflow saved while `disabled` existed (or any unrecognised value)
    // falls back to the default rather than leaving the widget blank.
    const stored = this.modeWidget && typeof this.modeWidget.value === "string"
      ? this.modeWidget.value
      : "";
    this.mode = MODES.includes(stored) ? stored : DEFAULT_MODE;
    if (this.modeWidget && this.modeWidget.value !== this.mode) {
      this.modeWidget.value = this.mode;
    }

    let data = null;
    const raw = this.rowsWidget ? this.rowsWidget.value : "";
    if (typeof raw === "string" && raw.trim()) {
      try {
        data = JSON.parse(raw);
      } catch (error) {
        console.warn("[F3Bbox] unreadable rows payload, starting from an empty table", error);
      }
    }
    // A bare array is also accepted (rows written by hand / by another node).
    if (Array.isArray(data)) data = { rows: data };
    if (!data || typeof data !== "object") data = { rows: [] };
    if (MODES.includes(data.mode)) {
      this.mode = data.mode;
      if (this.modeWidget) this.modeWidget.value = this.mode;
    }

    // Track which shape the payload used. In editing mode a null `tgt_bbox` is
    // meaningful (it is how FLUX.3 deletes an element), so it must survive the
    // round-trip; only a generation-mode row gets the bbox/tgt_bbox mirror.
    const loadedMode = data.mode === "editing" ? "editing" : "generation";
    this.rows = (Array.isArray(data.rows) ? data.rows : [])
      .filter((row) => row && typeof row === "object")
      .map((row) => ({
        id: String(row.id ?? "").trim(),
        desc: String(row.desc ?? ""),
        bbox: arrayToBox(row.bbox),
        from: typeof row.from === "string" && row.from.trim() ? row.from.trim() : null,
        src_bbox: arrayToBox(row.src_bbox),
        tgt_bbox: arrayToBox(row.tgt_bbox),
      }));

    // A generation row carries only bbox; keep both shapes populated so a mode
    // switch never loses the user's drawing. An editing row is authoritative.
    if (loadedMode === "generation") {
      for (const row of this.rows) {
        if (row.bbox && !row.tgt_bbox) row.tgt_bbox = clone(row.bbox);
        if (!row.bbox && row.tgt_bbox) row.bbox = clone(row.tgt_bbox);
      }
    }
    this.rows = this.rows.filter((row) => row.id);
    this.selected = this.rows.length ? 0 : -1;
    this._rebuildTable();
  }

  /**
   * Serialize the rows for the `bbox_rows` widget.
   *
   * In editing mode `bbox` mirrors `tgt_bbox`, including a null target for a
   * Remove row - writing the stale source box there instead would make the row
   * look like a Keep/Move on reload.
   */
  _serializeRows() {
    return this.rows.map((row) => {
      if (this.mode === "editing") {
        return {
          id: row.id,
          desc: row.desc,
          from: row.from || null,
          src_bbox: boxToArray(row.src_bbox),
          tgt_bbox: boxToArray(row.tgt_bbox),
          bbox: boxToArray(row.tgt_bbox),
        };
      }
      const box = boxToArray(row.bbox) || boxToArray(row.tgt_bbox);
      return {
        id: row.id,
        desc: row.desc,
        bbox: box,
        from: row.from || null,
        src_bbox: boxToArray(row.src_bbox),
        tgt_bbox: box,
      };
    });
  }

  /** Write the payload the Python node consumes. */
  _commit() {
    if (this.modeWidget) this.modeWidget.value = this.mode;

    if (this.rowsWidget) {
      this.rowsWidget.value = JSON.stringify({
        version: 1,
        mode: this.mode,
        rows: this._serializeRows(),
      });
    }
    // Mirrors so the table survives a round-trip even if the payload is missed.
    if (this.idsWidget) this.idsWidget.value = this.rows.map((r) => r.id).join("\n");
    if (this.descsWidget) this.descsWidget.value = JSON.stringify(this.rows.map((r) => r.desc));

    this._rebuildTable();
    this._render();
  }

  _captionText() {
    if (!this.rows.length) return "";
    const ids = this.rows.map((row) => `<${row.id}>`);
    return `Describe the image, naming each element where it appears: ${ids.join(", ")}.`;
  }

  async _copyCaption() {
    const text = this._captionText();
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      this._flashStatus("caption copied");
    } catch (error) {
      console.warn("[F3Bbox] clipboard unavailable", error);
      this._flashStatus("copy the caption from the console", true);
      console.log("[F3Bbox] caption:", text);
    }
  }

  /** One full-width labelled field line: `label [control]`. */
  _fieldLine(labelText, hint) {
    const wrap = document.createElement("label");
    wrap.style.cssText =
      "display:flex;align-items:flex-start;gap:6px;width:100%;box-sizing:border-box;";
    if (hint) wrap.title = hint;

    const label = document.createElement("span");
    label.textContent = labelText;
    label.style.cssText =
      `flex:0 0 ${FIELD_LABEL_PX}px;font:10px sans-serif;color:#8a8a8a;` +
      "padding-top:5px;text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
    wrap.appendChild(label);

    const slot = document.createElement("span");
    slot.style.cssText = "flex:1 1 auto;min-width:0;display:flex;";
    wrap.appendChild(slot);
    return { wrap, slot };
  }

  /** Select this row when a field is focused/clicked. */
  _bindRowFocus(control, index) {
    control.addEventListener("pointerdown", (e) => e.stopPropagation());
    control.addEventListener("focus", () => {
      this.selected = index;
      this._syncFieldButtons();
      this._render();
    });
  }

  /**
   * Row table. Each element is a card of full-width labelled fields:
   *
   *   editing      id / from ref / item desc, then the box readout + actions
   *   generation   id / desc, then the box readout + actions
   *
   * `item desc` deliberately sits *below* `from ref` in editing mode: the row reads
   * top-to-bottom as "which element -> where it comes from -> what it is/becomes".
   *
   * Deletion is two separate, clearly labelled actions, because they mean
   * different things:
   *   Remove      FLUX.3 element deletion. Keeps `from` + `src_bbox` and sets
   *               `tgt_bbox = null`, so the model erases that element from the
   *               image (the row still exports). Clicking again restores the
   *               target box.
   *   Delete row  drops the row from this editor only - it exports nothing. Use
   *               this to discard a mistake, not to delete an element.
   */
  _rebuildTable() {
    if (!this.table) return;
    this.table.innerHTML = "";

    // Empty state: say how to start instead of leaving a blank box.
    if (this.rows.length === 0) {
      const note = document.createElement("span");
      note.textContent = this.mode === "editing"
        ? "No rows yet - drag on the canvas to draw a box, then type its id, its from ref and its description."
        : "No rows yet - drag on the canvas to draw a box, then type its id and description.";
      note.style.cssText = "color:#999;padding:2px;";
      this.table.appendChild(note);
    }

    this.rows.forEach((row, index) => {
      const isRemove = this.mode === "editing" && !row.tgt_bbox;

      const card = document.createElement("div");
      card.style.cssText =
        "display:flex;flex-direction:column;gap:2px;width:100%;box-sizing:border-box;" +
        "padding:3px 5px 4px 6px;" +
        `border-left:3px solid ${this._colorFor(index)};background:${
          index === this.selected ? "#2a2a2a" : "transparent"};`;

      // ── header: index, row kind, actions ──────────────────────────────────
      const header = document.createElement("div");
      header.style.cssText =
        "display:flex;align-items:center;gap:5px;width:100%;box-sizing:border-box;";

      const num = document.createElement("span");
      num.textContent = `#${index + 1}`;
      num.style.cssText = "font:10px sans-serif;color:#888;flex:0 0 auto;";
      header.appendChild(num);

      const badge = document.createElement("span");
      badge.dataset.kindIndex = String(index);
      badge.style.cssText = "font:10px sans-serif;flex:0 0 auto;";
      header.appendChild(badge);

      const spacer = document.createElement("span");
      spacer.style.cssText = "flex:1 1 auto;";
      header.appendChild(spacer);

      // Deleting the element from the image (editing only: a generation row is
      // not taken from any image, so there is nothing to delete).
      if (this.mode === "editing") {
        const removeBtn = this._button(
          isRemove ? "Removing" : "Remove",
          "DELETE THIS ELEMENT FROM THE IMAGE.\n" +
          "Keeps where it came from (its from/source image and source box) but gives it no " +
          "target box, so the model erases it and fills the area with what was behind it.\n" +
          "Click again to bring the target box back. " +
          "(To merely drop this row from the editor, use 'Delete row' instead.)",
          () => this._toggleRemove(index),
        );
        removeBtn.style.flex = "0 0 auto";
        if (isRemove) {
          removeBtn.style.background = "#7a2b2b";
          removeBtn.style.color = "#fff";
          removeBtn.style.borderColor = "#a33";
        }
        header.appendChild(removeBtn);
      }

      const deleteBtn = this._button(
        "Delete row",
        "Remove this row from the editor table. This does NOT delete the element from the " +
        "image - use Remove for that.",
        () => {
          this.rows.splice(index, 1);
          this.selected = this.rows.length ? Math.min(index, this.rows.length - 1) : -1;
          this._commit();
        },
      );
      deleteBtn.style.flex = "0 0 auto";
      header.appendChild(deleteBtn);
      card.appendChild(header);

      // ── id ────────────────────────────────────────────────────────────────
      const idLine = this._fieldLine("id", "Element id. Reference it in the prompt as <id>.");
      const idInput = document.createElement("input");
      idInput.type = "text";
      idInput.value = row.id;
      idInput.placeholder = "element_1";
      idInput.style.cssText = INPUT_CSS;
      this._bindRowFocus(idInput, index);
      idInput.addEventListener("input", () => {
        row.id = idInput.value.trim();
        // Light commit: a full _commit() rebuilds the table and would steal
        // focus from this input after every keystroke.
        this._commitLight();
        this._updateStatus();
      });
      idLine.slot.appendChild(idInput);
      card.appendChild(idLine.wrap);

      // ── from ref (editing only) ───────────────────────────────────────────
      // This is the table's `from` field. `from`, `ref` and `src` all name the
      // SAME thing - the input image an element comes from - so the label spells
      // out both halves: `from` is the field, `ref_image_0` is its value.
      if (this.mode === "editing") {
        const fromLine = this._fieldLine(
          "from ref",
          "WHICH INPUT IMAGE this element comes from.\n" +
          "In the element table this is the `from` field, and its value is an image name: " +
          "ref_image_0 = the first image, ref_image_1 = the second.\n" +
          "`from`, `ref` and `src` all refer to this one thing: the source image. " +
          "`src_bbox` is simply the box inside that image.\n" +
          "Leave it EMPTY to say the element is not taken from any image - it is created " +
          "new inside the target box (that is what a New row is).",
        );
        const fromInput = document.createElement("input");
        fromInput.type = "text";
        fromInput.value = row.from || "";
        fromInput.placeholder = "ref_image_0  (empty = not from an image)";
        fromInput.style.cssText = INPUT_CSS;
        this._bindRowFocus(fromInput, index);
        fromInput.addEventListener("input", () => {
          const value = fromInput.value.trim();
          row.from = value ? value : null;
          // Light commit for the same reason as the id field: a rebuild here
          // would take focus away from the input being typed in.
          this._commitLight();
          this._updateStatus();
        });
        fromLine.slot.appendChild(fromInput);
        card.appendChild(fromLine.wrap);
      }

      // ── item desc (below from ref, as the row's final field) ─────────────
      // Labelled "item desc" in BOTH modes: `desc` describes the element, and it
      // applies to every row kind (Keep / Move / New / Remove) - calling it
      // "move desc" wrongly implied only Move rows have one. The mode-specific
      // meaning (editing = how it should look after the edit) lives in the
      // tooltip and the placeholder instead.
      const descLine = this._fieldLine(
        "item desc",
        "Element description, for every row kind. In editing mode this is what the " +
        "element should look like after the edit. Newlines are collapsed to one line " +
        "for the JSON table.",
      );
      const descInput = document.createElement("textarea");
      descInput.value = row.desc;
      descInput.rows = 2;
      descInput.placeholder = this.mode === "editing"
        ? "What the element should look like after the edit"
        : "What the element looks like";
      descInput.style.cssText = TEXTAREA_CSS;
      this._bindRowFocus(descInput, index);
      descInput.addEventListener("input", () => {
        row.desc = descInput.value;
        this._commitLight();
      });
      descLine.slot.appendChild(descInput);
      card.appendChild(descLine.wrap);

      // ── box readout ───────────────────────────────────────────────────────
      const boxInfo = document.createElement("span");
      boxInfo.dataset.rowIndex = String(index);
      boxInfo.style.cssText =
        `font:10px monospace;color:#999;padding-left:${FIELD_LABEL_PX + 6}px;` +
        "white-space:normal;word-break:break-word;";
      card.appendChild(boxInfo);

      card.addEventListener("pointerdown", () => {
        this.selected = index;
        this._syncFieldButtons();
        this._render();
      });

      this.table.appendChild(card);
    });
    this._updateStatus();
  }

  /**
   * Toggle the FLUX.3 "Remove" row for an element.
   *
   * Removing exports `{from, src_bbox, tgt_bbox: null}`: the model fills the
   * source box with what was behind the element. The target box the user had
   * drawn is remembered for the session so untoggling restores it exactly.
   */
  _toggleRemove(index) {
    const row = this.rows[index];
    if (!row) return;
    if (row.tgt_bbox) {
      row._removedTgt = clone(row.tgt_bbox);
      row.src_bbox = clone(row.src_bbox) || clone(row.bbox) || clone(row._removedTgt);
      row.tgt_bbox = null;
      // a Remove row must name a source, otherwise it is an empty New row
      row.from = row.from || "ref_image_0";
    } else {
      const restored = row._removedTgt || row.src_bbox || { t: 150, l: 150, b: 850, r: 850 };
      row.tgt_bbox = clone(restored);
      row.bbox = clone(restored);
      delete row._removedTgt;
    }
    this.selected = index;
    this.field = "tgt";
    this._commit();
  }

  /**
   * Refresh the per-row kind badge and box readout in place, without rebuilding
   * the inputs (which would steal focus from the field being typed in).
   */
  _updateStatus() {
    // Row-kind badge in each card header.
    for (const badge of this.table.querySelectorAll("span[data-kind-index]")) {
      const row = this.rows[Number(badge.dataset.kindIndex)];
      if (!row) continue;
      if (this.mode === "editing") {
        const kind = rowKind(row);
        badge.textContent = kind;
        badge.style.color = kind === "Remove" ? "#ff6b6b"
          : kind === "New" ? "#4cc9f0"
          : kind === "Keep" ? "#90be6d" : "#ffb703";
      } else {
        badge.textContent = "";
      }
    }

    // Box readout: source/target for editing, the single box otherwise.
    for (const span of this.table.querySelectorAll("span[data-row-index]")) {
      const row = this.rows[Number(span.dataset.rowIndex)];
      if (!row) continue;
      if (this.mode === "editing") {
        const src = boxToArray(row.src_bbox);
        const tgt = boxToArray(row.tgt_bbox);
        span.textContent =
          `src_bbox: ${src ? `[${src.join(", ")}]` : "null"}` +
          `   tgt_bbox: ${tgt ? `[${tgt.join(", ")}]` : "null (element removed)"}`;
        span.style.color = tgt ? "#999" : "#ff8f8f";
      } else {
        const box = boxToArray(row.bbox);
        span.textContent = box ? `bbox: [${box.join(", ")}]` : "bbox: no box yet";
        span.style.color = "#999";
      }
    }

    if (!this.status) return;
    const count = this.rows.length;
    const selected = this.rows[this.selected];
    const parts = [`${count} row${count === 1 ? "" : "s"}`];
    if (selected) {
      parts.push(`#${this.selected + 1} ${selected.id || "(no id)"}`);
      if (this.mode === "editing") parts.push(rowKind(selected));
      parts.push(`editing ${this.field === "src" ? "source" : "target"} box`);
    } else {
      parts.push("drag on the canvas to draw the first box");
    }
    this.status.textContent = parts.join(" - ");
  }

  _flashStatus(message, warn) {
    if (!this.status) return;
    const previous = this.status.textContent;
    this.status.style.color = warn ? "#ffb703" : "#90be6d";
    this.status.textContent = message;
    setTimeout(() => {
      this.status.style.color = "#8ab4f8";
      this.status.textContent = previous;
    }, 1500);
  }

  /**
   * Cheap commit while typing in a text input/textarea: rewrites the payload but
   * never rebuilds the table, so the field being typed in keeps focus. The
   * in-memory rows are already correct; only the serialized payload is stale.
   */
  _commitLight() {
    if (this.rowsWidget) {
      this.rowsWidget.value = JSON.stringify({
        version: 1,
        mode: this.mode,
        rows: this._serializeRows(),
      });
    }
    if (this.idsWidget) this.idsWidget.value = this.rows.map((r) => r.id).join("\n");
    if (this.descsWidget) this.descsWidget.value = JSON.stringify(this.rows.map((r) => r.desc));
  }

  // ── geometry ─────────────────────────────────────────────────────────────

  get _W() {
    return Math.max(1, this._sourceWidth || 1024);
  }

  get _H() {
    return Math.max(1, this._sourceHeight || 1024);
  }

  /** Grid box -> canvas px. */
  _boxToCanvas(box) {
    const n = normalizeBox(box);
    if (!n) return null;
    return {
      x0: (n.l / GRID) * this._W * this.scaleX,
      y0: (n.t / GRID) * this._H * this.scaleY,
      x1: (n.r / GRID) * this._W * this.scaleX,
      y1: (n.b / GRID) * this._H * this.scaleY,
    };
  }

  /** Canvas px -> grid box. */
  _canvasToBox(x0, y0, x1, y1) {
    return normalizeBox({
      t: (Math.min(y0, y1) / this.scaleY / this._H) * GRID,
      l: (Math.min(x0, x1) / this.scaleX / this._W) * GRID,
      b: (Math.max(y0, y1) / this.scaleY / this._H) * GRID,
      r: (Math.max(x0, x1) / this.scaleX / this._W) * GRID,
    });
  }

  /** The box a canvas drag currently edits for a row. */
  _activeBox(row) {
    if (!row) return null;
    if (this.mode === "editing") {
      return this.field === "src" ? row.src_bbox : (row.tgt_bbox || row.bbox);
    }
    return row.bbox || row.tgt_bbox;
  }

  _setActiveBox(row, box) {
    if (!row) return;
    if (this.mode === "editing") {
      if (this.field === "src") {
        row.src_bbox = box;
      } else {
        row.tgt_bbox = box;
        row.bbox = box;
      }
    } else {
      row.bbox = box;
      row.tgt_bbox = box;
    }
  }

  _colorFor(index) {
    return ROW_COLORS[index % ROW_COLORS.length];
  }

  // ── layout ───────────────────────────────────────────────────────────────

  _measureControls(reflow) {
    const measured = this.controls ? this.controls.offsetHeight : 0;
    if (measured > 0) {
      const next = Math.ceil(measured) + 4;
      if (next !== this.node._f3ControlsH) {
        this.node._f3ControlsH = next;
        if (reflow) this._resizeNode();
      }
    }
  }

  _relayout() {
    const avail = Math.max(140, Math.round((this.node.size ? this.node.size[0] : 430) - 45));
    let displayW = Math.min(avail, MAX_DISPLAY);
    let displayH = Math.max(1, Math.round(displayW * (this._H / this._W)));
    if (displayH > MAX_DISPLAY) {
      const k = MAX_DISPLAY / displayH;
      displayH = MAX_DISPLAY;
      displayW = Math.max(1, Math.round(displayW * k));
    }
    this.displayW = displayW;
    this.displayH = displayH;
    this.scaleX = displayW / this._W;
    this.scaleY = displayH / this._H;

    this.canvas.width = displayW + MARGIN * 2;
    this.canvas.height = displayH + MARGIN * 2;
    this.canvas.style.width = `${this.canvas.width}px`;
    this.canvas.style.height = `${this.canvas.height}px`;
    this.node._f3CanvasH = this.canvas.height;

    this._resizeNode();
    this._render();
  }

  _resizeNode() {
    const total = this.node._f3CanvasH + this.node._f3ControlsH + this._chromeHeight();
    const width = Math.max(this.node.size[0], 320);
    // Skip a no-op setSize: it would fire onResize -> _relayout -> _resizeNode
    // again, and the repeated layout passes below would otherwise churn.
    if (Math.abs(this.node.size[0] - width) < 0.5 && Math.abs(this.node.size[1] - total) < 0.5) return;
    this.node.setSize([width, total]);
    if (this.node.graph) this.node.graph.setDirtyCanvas(true, true);
  }

  /**
   * Force a full re-measure and repaint of the canvas + controls.
   *
   * ComfyUI finalises a node's size and the DOM widgets' box asynchronously
   * after onNodeCreated, so the first layout can be computed against a stale
   * node width and the canvas ends up blank until something else triggers a
   * redraw. This is also wired to the `Refresh` button so the user never has to
   * nudge the node (adding and deleting a box) to make it appear.
   *
   * Non-destructive: it never re-reads `bbox_rows`, so the rows and the current
   * selection survive.
   */
  refreshLayout() {
    this._readWidgets();
    this._measureControls(false);
    this._relayout();
    this._syncFromConnectedImage();
  }

  /** Height taken by the title bar plus the node's own visible value widgets. */
  _chromeHeight() {
    let rows = 0;
    for (const widget of this.node.widgets || []) {
      if (widget === this.node._f3CanvasWidget || widget === this.node._f3ControlsWidget) continue;
      if (widget.hidden) continue;
      rows += 1;
    }
    return 34 + rows * 44 + 16;
  }

  // ── events ───────────────────────────────────────────────────────────────

  _setupEvents() {
    this._onPointerDown = (e) => this.onPointerDown(e);
    this._onPointerMove = (e) => this.onPointerMove(e);
    this._onPointerUp = (e) => this.onPointerUp(e);
    this._onContextMenu = (e) => e.preventDefault();
    this._onPointerLeave = () => {
      if (!this.drag) this.canvas.style.cursor = "crosshair";
    };

    this.canvas.addEventListener("pointerdown", this._onPointerDown);
    this.canvas.addEventListener("pointermove", this._onPointerMove);
    this.canvas.addEventListener("pointerup", this._onPointerUp);
    this.canvas.addEventListener("pointercancel", this._onPointerUp);
    this.canvas.addEventListener("contextmenu", this._onContextMenu);
    this.canvas.addEventListener("pointerleave", this._onPointerLeave);
  }

  destroy() {
    // Stop any pending delayed layout pass from touching a removed node.
    this.destroyed = true;
    this.canvas.removeEventListener("pointerdown", this._onPointerDown);
    this.canvas.removeEventListener("pointermove", this._onPointerMove);
    this.canvas.removeEventListener("pointerup", this._onPointerUp);
    this.canvas.removeEventListener("pointercancel", this._onPointerUp);
    this.canvas.removeEventListener("contextmenu", this._onContextMenu);
    this.canvas.removeEventListener("pointerleave", this._onPointerLeave);
  }

  _mouse(e) {
    const rect = this.canvas.getBoundingClientRect();
    const sx = this.canvas.width / Math.max(rect.width, 1);
    const sy = this.canvas.height / Math.max(rect.height, 1);
    const raw = { x: (e.clientX - rect.left) * sx, y: (e.clientY - rect.top) * sy };
    return { x: raw.x - MARGIN, y: raw.y - MARGIN, rawX: raw.x, rawY: raw.y };
  }

  /** Corner handles of the selected row's active box. */
  _handles(box) {
    const r = this._boxToCanvas(box);
    if (!r) return [];
    return [
      { id: "nw", x: r.x0, y: r.y0 },
      { id: "ne", x: r.x1, y: r.y0 },
      { id: "sw", x: r.x0, y: r.y1 },
      { id: "se", x: r.x1, y: r.y1 },
    ];
  }

  _hitHandle(mouse) {
    const row = this.rows[this.selected];
    if (!row) return null;
    for (const handle of this._handles(this._activeBox(row))) {
      if (Math.hypot(handle.x - mouse.x, handle.y - mouse.y) <= HANDLE + 3) return handle.id;
    }
    return null;
  }

  /** Topmost row whose active box contains the point. */
  _hitBox(mouse) {
    for (let i = this.rows.length - 1; i >= 0; i--) {
      const r = this._boxToCanvas(this._activeBox(this.rows[i]));
      if (!r) continue;
      const pad = 2;
      if (mouse.x >= r.x0 - pad && mouse.x <= r.x1 + pad
        && mouse.y >= r.y0 - pad && mouse.y <= r.y1 + pad) return i;
    }
    return -1;
  }

  onPointerDown(e) {
    e.preventDefault();
    e.stopPropagation();

    const mouse = this._mouse(e);
    const inside = mouse.x >= -MARGIN && mouse.y >= -MARGIN
      && mouse.x <= this.displayW + MARGIN && mouse.y <= this.displayH + MARGIN;

    if (e.button === 2) {
      // Right click deletes the row under the cursor.
      const hit = this._hitBox(mouse);
      if (hit >= 0) {
        this.rows.splice(hit, 1);
        this.selected = this.rows.length ? Math.min(hit, this.rows.length - 1) : -1;
        this._commit();
      }
      return;
    }
    if (e.button !== 0 || !inside) return;

    const handle = this._hitHandle(mouse);
    if (handle) {
      const row = this.rows[this.selected];
      this.drag = {
        mode: "resize", handle, index: this.selected,
        startBox: clone(this._activeBox(row)), startMouse: mouse,
      };
      this.canvas.setPointerCapture?.(e.pointerId);
      return;
    }

    const hit = this._hitBox(mouse);
    if (hit >= 0 && !e.shiftKey) {
      this.selected = hit;
      this._syncFieldButtons();
      const row = this.rows[hit];
      const startBox = clone(this._activeBox(row));
      this.drag = {
        mode: "move", index: hit, startBox, startMouse: mouse,
      };
      this.canvas.setPointerCapture?.(e.pointerId);
      this.canvas.style.cursor = "move";
      this._render();
      return;
    }

    // Empty canvas (or shift+drag): draw a new box for the selected row, or
    // create a row when there is none. The new row must show up in the table
    // right away, so this commits (a rebuild) rather than the light commit.
    if (this.rows.length === 0 || this.selected < 0) {
      this._addRow_(true, mouse);
    } else {
      this._commit();
    }
    const row = this.rows[this.selected];
    if (!row) return;
    this.drag = {
      mode: "draw", index: this.selected, startMouse: mouse,
      startBox: clone(this._activeBox(row)),
    };
    this.canvas.setPointerCapture?.(e.pointerId);
    this.canvas.style.cursor = "crosshair";
    // Draw immediately so a click without movement still yields a box.
    const box = this._canvasToBox(mouse.x, mouse.y, mouse.x, mouse.y);
    this._setActiveBox(row, box);
    this._commitLight();
    this._render();
  }

  onPointerMove(e) {
    const mouse = this._mouse(e);
    if (!this.drag) {
      if (this._hitHandle(mouse)) {
        this.canvas.style.cursor = "nwse-resize";
      } else if (this._hitBox(mouse) >= 0) {
        this.canvas.style.cursor = "move";
      } else {
        this.canvas.style.cursor = "crosshair";
      }
      this._hoverInfo(mouse);
      return;
    }

    const row = this.rows[this.drag.index];
    if (!row) return;
    e.stopPropagation();

    if (this.drag.mode === "draw") {
      this._setActiveBox(row, this._canvasToBox(
        this.drag.startMouse.x, this.drag.startMouse.y, mouse.x, mouse.y));
    } else if (this.drag.mode === "move") {
      const start = this.drag.startBox;
      if (!start) return;
      const dxGrid = ((mouse.x - this.drag.startMouse.x) / this.scaleX / this._W) * GRID;
      const dyGrid = ((mouse.y - this.drag.startMouse.y) / this.scaleY / this._H) * GRID;
      const moved = normalizeBox({
        t: start.t + dyGrid, l: start.l + dxGrid,
        b: start.b + dyGrid, r: start.r + dxGrid,
      });
      // Clamp translation so the box cannot be dragged off the frame.
      const w = moved.r - moved.l;
      const h = moved.b - moved.t;
      moved.l = clamp(moved.l, 0, GRID - w);
      moved.r = moved.l + w;
      moved.t = clamp(moved.t, 0, GRID - h);
      moved.b = moved.t + h;
      this._setActiveBox(row, moved);
    } else if (this.drag.mode === "resize") {
      const r = this._boxToCanvas(this.drag.startBox);
      if (!r) return;
      let x0 = r.x0, y0 = r.y0, x1 = r.x1, y1 = r.y1;
      if (this.drag.handle.includes("w")) x0 = mouse.x;
      if (this.drag.handle.includes("e")) x1 = mouse.x;
      if (this.drag.handle.includes("n")) y0 = mouse.y;
      if (this.drag.handle.includes("s")) y1 = mouse.y;
      this._setActiveBox(row, this._canvasToBox(x0, y0, x1, y1));
    }
    this._commitLight();
    this._render();
    this._updateStatus();
  }

  onPointerUp(e) {
    if (!this.drag) return;
    // A click that never moved would commit a zero-area box, which the Python
    // side rejects. Drop a usable default box instead so the row is valid.
    const row = this.rows[this.drag.index];
    if (row && this.drag.mode === "draw") {
      const r = this._boxToCanvas(this._activeBox(row));
      if (r && (r.x1 - r.x0) < MIN_BOX_PX && (r.y1 - r.y0) < MIN_BOX_PX) {
        const start = this.drag.startMouse;
        const w = 500;
        const h = 500;
        const gl = clamp((start.x / this.scaleX / this._W) * GRID - w / 2, 0, GRID - w);
        const gt = clamp((start.y / this.scaleY / this._H) * GRID - h / 2, 0, GRID - h);
        this._setActiveBox(row, normalizeBox({ t: gt, l: gl, b: gt + h, r: gl + w }));
      }
    }
    this.drag = null;
    this.canvas.style.cursor = "crosshair";
    this._commit();
  }

  _hoverInfo(mouse) {
    if (!this.status) return;
    const gx = Math.round(clamp((mouse.x / this.scaleX / this._W) * GRID, 0, GRID));
    const gy = Math.round(clamp((mouse.y / this.scaleY / this._H) * GRID, 0, GRID));
    const base = this.rows.length
      ? `${this.rows.length} row${this.rows.length === 1 ? "" : "s"} - cursor (l,t) ${gx},${gy}`
      : "drag to draw the first box";
    this.status.textContent = base;
  }

  // ── rows ─────────────────────────────────────────────────────────────────

  /**
   * Append an element row.
   * @param {boolean} commit  write the payload immediately
   * @param {object=} mouse   canvas position to place the initial box at
   */
  _addRow_(commit = true, mouse) {
    const index = this.rows.length;
    void index;
    let box = null;
    if (mouse) {
      const gx = clamp((mouse.x / this.scaleX / this._W) * GRID, 0, GRID - 100);
      const gy = clamp((mouse.y / this.scaleY / this._H) * GRID, 0, GRID - 100);
      box = normalizeBox({ t: gy, l: gx, b: gy + 100, r: gx + 100 });
    }
    const row = {
      id: suggestId("element"),
      desc: "",
      bbox: box,
      src_bbox: this.mode === "editing" ? clone(box) : null,
      tgt_bbox: clone(box),
      from: this.mode === "editing" ? "ref_image_0" : null,
    };
    this.rows.push(row);
    this.selected = this.rows.length - 1;
    if (commit) this._commit();
    return row;
  }

  _addBackgroundRow() {
    const row = this._addRow_(false);
    const full = { t: 0, l: 0, b: GRID, r: GRID };
    row.bbox = clone(full);
    row.tgt_bbox = clone(full);
    row.src_bbox = this.mode === "editing" ? clone(full) : null;
    row.desc = row.desc || "The whole scene.";
    this._commit();
  }

  /** Set from/src_bbox/tgt_bbox to match a row kind, keeping the drawn boxes. */
  _applyKind(kind) {
    const row = this.rows[this.selected];
    if (!row) return;
    const tgt = clone(row.tgt_bbox || row.bbox || { t: 150, l: 150, b: 850, r: 850 });
    const src = clone(row.src_bbox || row.bbox || tgt);
    if (kind === "Keep") {
      row.from = row.from || "ref_image_0";
      row.src_bbox = clone(src);
      row.tgt_bbox = clone(src);
    } else if (kind === "Move") {
      row.from = row.from || "ref_image_0";
      row.src_bbox = clone(src);
      row.tgt_bbox = clone(tgt);
      if (boxToArray(row.src_bbox)?.join() === boxToArray(row.tgt_bbox)?.join()) {
        // nudge the target so the row is genuinely a Move
        const moved = normalizeBox({
          t: max0(tgt.t - 80), l: max0(tgt.l - 80), b: tgt.b, r: tgt.r,
        });
        row.tgt_bbox = moved;
      }
    } else if (kind === "New") {
      row.from = null;
      row.src_bbox = null;
      row.tgt_bbox = clone(tgt);
    } else if (kind === "Remove") {
      row.from = row.from || "ref_image_0";
      row.src_bbox = clone(src);
      row.tgt_bbox = null;
    }
    row.bbox = clone(row.tgt_bbox || row.src_bbox);
    this.field = kind === "Remove" ? "src" : "tgt";
    this._commit();
  }

  // ── image underlay ───────────────────────────────────────────────────────

  _syncFromConnectedImage() {
    try {
      const graph = this.node.graph;
      if (!graph) return;
      const input = (this.node.inputs || []).find((i) => i.name === "image");
      if (!input || input.link == null) return;
      const link = graph.links instanceof Map ? graph.links.get(input.link) : graph.links?.[input.link];
      const origin = link ? graph.getNodeById?.(link.origin_id) : null;
      if (!origin) return;
      const fileWidget = (origin.widgets || []).find((w) => w.name === "image" || w.name === "video");
      if (!fileWidget || typeof fileWidget.value !== "string" || !fileWidget.value) return;
      const cleaned = fileWidget.value.replace(/\s*\[(input|output|temp)\]\s*$/i, "");
      const url = `/view?filename=${encodeURIComponent(cleaned)}&type=input`;
      const img = new Image();
      img.onload = () => {
        this.guideImg = img;
        this._sourceWidth = img.naturalWidth || 1024;
        this._sourceHeight = img.naturalHeight || 1024;
        this._relayout();
      };
      img.onerror = () => { /* no preview available; the grid still works */ };
      img.src = url;
      this._watchSourceWidget(fileWidget);
    } catch (error) {
      console.warn("[F3Bbox] could not read the connected image", error);
    }
  }

  _watchSourceWidget(widget) {
    if (this._watchedWidget === widget) return;
    const original = widget.callback;
    this._watchedWidget = widget;
    this._watchedCallback = original;
    widget.callback = (...args) => {
      const result = typeof original === "function" ? original.apply(widget, args) : undefined;
      setTimeout(() => this._syncFromConnectedImage(), 50);
      return result;
    };
  }

  refresh() {
    this._readWidgets();
    this._readState();
    this._relayout();
  }

  // ── render ───────────────────────────────────────────────────────────────

  _render() {
    const ctx = this.ctx;
    const W = this.canvas.width;
    const H = this.canvas.height;
    ctx.clearRect(0, 0, W, H);

    // checkerboard so a transparent/black reference is still readable
    ctx.save();
    ctx.fillStyle = "#1a1a1a";
    ctx.fillRect(0, 0, W, H);
    ctx.translate(MARGIN, MARGIN);

    const dw = this.displayW || 1;
    const dh = this.displayH || 1;

    ctx.fillStyle = "#0f0f0f";
    ctx.fillRect(0, 0, dw, dh);

    if (this.guideEnabled && this.guideImg) {
      ctx.globalAlpha = 0.85;
      try {
        ctx.drawImage(this.guideImg, 0, 0, dw, dh);
      } catch (error) {
        /* ignore draw errors */
      }
      ctx.globalAlpha = 1;
    }

    // 10x10 grid: the FLUX.3 coordinate grid the boxes live on
    ctx.strokeStyle = "rgba(255,255,255,0.10)";
    ctx.lineWidth = 1;
    for (let i = 1; i < 10; i++) {
      const x = (dw / 10) * i;
      const y = (dh / 10) * i;
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, dh); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(dw, y); ctx.stroke();
    }
    // half grid lines (20 divisions) for finer placement
    ctx.strokeStyle = "rgba(255,255,255,0.05)";
    for (let i = 1; i < 20; i++) {
      if (i % 2 === 0) continue;
      const x = (dw / 20) * i;
      const y = (dh / 20) * i;
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, dh); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(dw, y); ctx.stroke();
    }

    ctx.strokeStyle = "rgba(255,255,255,0.35)";
    ctx.strokeRect(0.5, 0.5, dw - 1, dh - 1);

    // Empty state: a nudge to start drawing rather than a blank frame.
    if (this.rows.length === 0) {
      ctx.fillStyle = "rgba(200,200,200,0.75)";
      ctx.font = "12px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("drag here to draw the first box", dw / 2, dh / 2);
    }

    this.rows.forEach((row, index) => {
      const color = this._colorFor(index);
      const selected = index === this.selected;

      if (this.mode === "editing") {
        // source box: where the element is now (drawn only when it has one)
        if (row.src_bbox) {
          this._drawBox(row.src_bbox, SRC_COLOR, selected, false, `${row.id} src`);
        }
        // target box: where it should end up (dashed); absent = Remove row
        if (row.tgt_bbox) {
          this._drawBox(row.tgt_bbox, TGT_COLOR, selected, true, `${row.id} tgt`);
        } else if (row.src_bbox) {
          const r = this._boxToCanvas(row.src_bbox);
          if (r) {
            ctx.fillStyle = "rgba(255,80,80,0.85)";
            ctx.font = "bold 11px sans-serif";
            ctx.textAlign = "center";
            ctx.fillText("REMOVE", (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2);
          }
        }
      } else if (row.bbox) {
        this._drawBox(row.bbox, color, selected, false, row.id);
      }
    });

    // Handles for the selected row's active box
    const selectedRow = this.rows[this.selected];
    if (selectedRow) {
      for (const handle of this._handles(this._activeBox(selectedRow))) {
        ctx.fillStyle = "#ffffff";
        ctx.strokeStyle = "#000000";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.rect(handle.x - HANDLE, handle.y - HANDLE, HANDLE * 2, HANDLE * 2);
        ctx.fill();
        ctx.stroke();
      }
    }

    // Guide/caption hint
    ctx.fillStyle = "rgba(160,160,160,0.8)";
    ctx.font = "10px sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("boxes: [top, left, bottom, right] on a 0-1000 grid", 2, dh + 11);

    ctx.restore();
  }

  _drawBox(box, color, selected, dashed, label) {
    const ctx = this.ctx;
    const r = this._boxToCanvas(box);
    if (!r) return;
    const w = r.x1 - r.x0;
    const h = r.y1 - r.y0;

    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = selected ? 2.5 : 1.5;
    if (dashed) ctx.setLineDash([6, 4]);
    ctx.strokeRect(r.x0, r.y0, w, h);
    ctx.setLineDash([]);

    ctx.fillStyle = this._withAlpha(color, selected ? 0.16 : 0.09);
    ctx.fillRect(r.x0, r.y0, w, h);

    if (this.showLabels && label) {
      ctx.font = `${selected ? "bold " : ""}11px sans-serif`;
      const textW = ctx.measureText(label).width;
      const pad = 3;
      const ly = Math.max(11, r.y0 - 3);
      ctx.fillStyle = "rgba(0,0,0,0.72)";
      ctx.fillRect(r.x0, ly - 11, textW + pad * 2, 14);
      ctx.fillStyle = color;
      ctx.textAlign = "left";
      ctx.fillText(label, r.x0 + pad, ly);
    }
    ctx.restore();
  }

  _withAlpha(hex, alpha) {
    const m = /^#([0-9a-f]{6})$/i.exec(hex);
    if (!m) return `rgba(255,255,255,${alpha})`;
    const n = parseInt(m[1], 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${alpha})`;
  }
}

// ── registration ───────────────────────────────────────────────────────────

async function getApp() {
  if (window.comfyAPI && window.comfyAPI.app && window.comfyAPI.app.app) {
    return window.comfyAPI.app.app;
  }
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
      name: "EditUtils.Flux3BboxEditor",
      async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!nodeData || nodeData.name !== NODE_NAME) return;

        chainCallback(nodeType.prototype, "onNodeCreated", function () {
          if (this._f3Editor) this._f3Editor.destroy();
          this._f3Editor = new Flux3BboxEditor(this);
        });

        chainCallback(nodeType.prototype, "onConfigure", function () {
          const editor = this._f3Editor;
          if (!editor) return;
          // Widget values are restored before onConfigure runs, so the rows and
          // the canvas size have to be taken from them again. The extra pass
          // runs after the frontend has finished sizing the node and its DOM
          // widgets - that is the point where a freshly LOADED workflow used to
          // show a blank canvas.
          setTimeout(() => editor.refresh(), 0);
          setTimeout(() => {
            if (!editor.destroyed) editor.refreshLayout();
          }, 120);
        });

        chainCallback(nodeType.prototype, "onResize", function () {
          if (this._f3Editor) this._f3Editor._relayout();
        });

        chainCallback(nodeType.prototype, "onConnectionsChange", function () {
          if (this._f3Editor) setTimeout(() => this._f3Editor._syncFromConnectedImage(), 50);
        });

        chainCallback(nodeType.prototype, "onRemoved", function () {
          if (this._f3Editor) {
            this._f3Editor.destroy();
            this._f3Editor = null;
          }
        });
      },
    });
  })
  .catch((error) => console.error("[F3Bbox] failed to register the bbox editor", error));
