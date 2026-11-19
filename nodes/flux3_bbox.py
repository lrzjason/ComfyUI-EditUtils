"""FLUX.3 bounding-box element table: parsing/serialising helpers and the canvas node."""

import math
import json
import copy

from ._shared import stamp_config_id

FLUX3_BBOX_GRID = 1000.0


def parse_flux3_bbox(value, field="bbox"):
    """Coerce a bbox to the FLUX.3 [top, left, bottom, right] int contract.

    Accepts a list/tuple of 4 numbers, or the string form the canvas writes
    ("[150, 150, 850, 850]"). Values are clamped to 0-1000 and rounded, and the
    box is normalized so top <= bottom and left <= right (a box dragged
    bottom-right to top-left must not produce an inverted row the model would
    read as empty). Returns None when the value is null/"null"/absent/invalid —
    which is meaningful in the edit table (src_bbox null = New row, tgt_bbox
    null = Remove row).
    """
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() == "null":
            return None
        try:
            value = json.loads(text)
        except Exception:
            # also accept "t,l,b,r" without brackets
            try:
                value = [float(part) for part in text.replace("[", "").replace("]", "").split(",")]
            except Exception:
                print(f"Flux3Klein bbox: could not parse {field}={text!r}, treating as null")
                return None
    if isinstance(value, dict):
        # tolerate {"t":..,"l":..,"b":..,"r":..}
        try:
            value = [value["t"], value["l"], value["b"], value["r"]]
        except Exception:
            print(f"Flux3Klein bbox: unusable dict for {field}, treating as null")
            return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        print(f"Flux3Klein bbox: {field} needs exactly 4 numbers, got {value!r}")
        return None
    try:
        nums = [float(v) for v in value]
    except Exception:
        print(f"Flux3Klein bbox: {field} has a non-numeric value in {value!r}")
        return None
    if any(math.isnan(n) for n in nums):
        return None

    t, l, b, r = nums
    # normalize ordering, then clamp to the grid and round to integers
    t, b = min(t, b), max(t, b)
    l, r = min(l, r), max(l, r)
    box = [
        int(round(max(0.0, min(FLUX3_BBOX_GRID, t)))),
        int(round(max(0.0, min(FLUX3_BBOX_GRID, l)))),
        int(round(max(0.0, min(FLUX3_BBOX_GRID, b)))),
        int(round(max(0.0, min(FLUX3_BBOX_GRID, r)))),
    ]
    if box[0] == box[2] or box[1] == box[3]:
        # a zero-area box cannot be a layout target
        print(f"Flux3Klein bbox: {field} {box} has zero area, treating as null")
        return None
    return box


def _flux3_clean_desc(value):
    """One-line-safe description text for an element row."""
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    # The table is JSON, so newlines are legal inside a string, but a stray
    # newline in a hand-edited widget usually means "two rows"; collapse to a
    # single line to keep the prompt compact and predictable.
    return " ".join(part.strip() for part in text.split("\n") if part.strip()).strip()


def parse_flux3_bbox_payload(payload, width=None, height=None):
    """Parse the canvas payload into normalized element rows.

    The canvas writes:
        {"version": 1, "mode": "generation"|"editing",
         "rows": [{"id": "background_1", "desc": "...",
                   "bbox": [t,l,b,r], "from": "ref_image_0",
                   "src_bbox": [...], "tgt_bbox": [...]}]}

    Rows are returned with the FLUX.3 field names for the active mode, plus the
    originating reference index when a "ref_image_N" was named.
    """
    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            return [], "generation"
        try:
            payload = json.loads(text)
        except Exception as e:
            raise ValueError(f"Flux3Klein bbox rows: invalid JSON ({e})") from e
    if payload is None:
        return [], "generation"
    if isinstance(payload, list):
        # bare array of rows / of bboxes
        payload = {"rows": payload}
    if not isinstance(payload, dict):
        raise ValueError("Flux3Klein bbox rows: expected a JSON object or array")

    mode = str(payload.get("mode", "generation")).lower()
    if mode not in ("generation", "editing"):
        print(f"Flux3Klein bbox: unknown mode {mode!r}, using generation")
        mode = "generation"

    raw_rows = payload.get("rows")
    if raw_rows is None:
        return [], mode
    if not isinstance(raw_rows, list):
        raise ValueError("Flux3Klein bbox rows: 'rows' must be a list")

    rows = []
    seen_ids = set()
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, dict):
            print(f"Flux3Klein bbox: row {index} is not an object, skipping")
            continue

        element_id = str(raw.get("id", "") or "").strip()
        if not element_id:
            print(f"Flux3Klein bbox: row {index} has no id, skipping")
            continue

        desc = _flux3_clean_desc(raw.get("desc", ""))

        # 'from' identifies the source reference in editing mode.
        from_ref = raw.get("from", None)
        if isinstance(from_ref, str):
            from_ref = from_ref.strip()
            if not from_ref or from_ref.lower() == "null":
                from_ref = None

        if mode == "generation":
            bbox = parse_flux3_bbox(raw.get("bbox", raw.get("tgt_bbox")), "bbox")
            if bbox is None:
                print(f"Flux3Klein bbox: generation row {element_id!r} has no usable bbox, skipping")
                continue
            row = {"id": element_id, "bbox": bbox, "desc": desc}
        else:
            src = parse_flux3_bbox(raw.get("src_bbox"), "src_bbox")
            tgt = parse_flux3_bbox(raw.get("tgt_bbox", raw.get("bbox")), "tgt_bbox")
            # A New row generates the element: no source, only a target.
            if from_ref is None:
                src = None
            if src is None and tgt is None:
                print(f"Flux3Klein bbox: editing row {element_id!r} has neither src_bbox nor tgt_bbox, skipping")
                continue
            if from_ref is not None and src is None and tgt is not None:
                # from set but no source box: that is not a defined row kind, so
                # treat it as a New row rather than emitting an invalid row.
                print(f"Flux3Klein bbox: row {element_id!r} has 'from' but no src_bbox; treating as a New row")
                from_ref = None
            row = {
                "id": element_id,
                "from": from_ref,
                "src_bbox": src,
                "tgt_bbox": tgt,
                "desc": desc,
            }
            # Keep / Move / New / Remove classify the row for logging only; the
            # model reads the fields themselves.
            if from_ref is None:
                row_kind = "New"
            elif tgt is None:
                row_kind = "Remove"
            elif src == tgt:
                row_kind = "Keep"
            else:
                row_kind = "Move"
            row["_kind"] = row_kind

        if element_id in seen_ids:
            print(f"Flux3Klein bbox: duplicate id {element_id!r}; the prompt references it once, "
                  "so later rows with the same id will be ambiguous")
        seen_ids.add(element_id)
        rows.append(row)

    return rows, mode


def _flux3_row_is_editing(row):
    """True when a parsed row carries the editing fields.

    Parsed rows are unambiguous: a generation row is exactly {id, bbox, desc},
    while an editing row always has the `from` key (its value may be None).
    """
    return any(key in row for key in ("from", "src_bbox", "tgt_bbox"))


def build_flux3_bbox_json(rows, mode="generation"):
    """Serialize element rows into the compact JSON array FLUX.3 expects.

    Each row is emitted in the schema it was actually drawn in rather than in one
    global ``mode``. A chain can mix a generation table on one image with an
    editing table on another, and forcing a single schema onto the merge would
    either crash (a generation row has no ``bbox``) or silently emit the wrong
    fields for some rows. ``mode`` is kept for call-site compatibility and no
    longer selects the shape.

    The internal '_kind' bookkeeping key is dropped so the emitted JSON contains
    exactly the documented fields.
    """
    out = []
    for row in rows:
        if _flux3_row_is_editing(row):
            out.append({
                "id": row["id"],
                "from": row.get("from"),
                "src_bbox": list(row["src_bbox"]) if row.get("src_bbox") else None,
                "tgt_bbox": list(row["tgt_bbox"]) if row.get("tgt_bbox") else None,
                "desc": row.get("desc", ""),
            })
        else:
            box = row.get("bbox") or row.get("tgt_bbox")
            if not box:
                # Unusable row: skip it rather than raising inside the encoder.
                continue
            out.append({
                "id": row["id"],
                "bbox": list(box),
                "desc": row.get("desc", ""),
            })
    return json.dumps(out, ensure_ascii=False)


def compose_flux3_prompt(instruction, rows, mode="generation"):
    """Append the element table to the instruction, FLUX.3 style.

    The documented format is "<caption/instruction> <space> <JSON array>": the
    table goes last, in the same prompt string, with no separate parameter.
    """
    table = build_flux3_bbox_json(rows, mode)
    base = (instruction or "").strip()
    if not rows:
        return base
    return f"{base} {table}".strip() if base else table


def describe_flux3_bbox_rows(rows, mode="generation"):
    """Human-readable one-line-per-row summary, for node logging.

    Describes each row from its own fields, so a chain that mixes generation and
    editing rows is logged correctly (the old mode-based version indexed
    ``r['bbox']`` on editing rows and raised).
    """
    if not rows:
        return "no bbox rows"
    parts = []
    for row in rows:
        if _flux3_row_is_editing(row):
            parts.append(
                f"{row['id']}({row.get('_kind', '?')}=src:{row.get('src_bbox')}->tgt:{row.get('tgt_bbox')})"
            )
        else:
            box = row.get("bbox") or row.get("tgt_bbox")
            parts.append(f"{row['id']}=[{' '.join(map(str, box)) if box else '-'}]")
    return "; ".join(parts)


def collect_flux3_bbox_rows(configs):
    """Merge the element-table rows of a config chain.

    Walks the configs in order and concatenates every config's parsed rows,
    exactly the way ``EditTextEncode_EditUtils`` builds the prompt table, so
    the JSON a preparer node reports is the JSON that reaches the model. The
    mode comes from the last config that actually contributed rows (the
    encoder's rule when a chain mixes modes).

    Returns ``(rows, mode)``.
    """
    merged_rows = []
    merged_mode = "generation"
    for image_obj in configs or []:
        if not isinstance(image_obj, dict):
            continue
        rows = image_obj.get("bbox_rows_parsed") or []
        if not rows:
            continue
        merged_mode = image_obj.get("bbox_mode_parsed", "generation")
        merged_rows.extend(rows)
    return merged_rows, merged_mode


def flux3_bbox_rows_json(configs):
    """The element-table JSON array for a config chain.

    Returns an **empty string** (not ``"[]"``) when there are no rows, so a
    downstream text node shows nothing rather than an empty array.
    """
    rows, mode = collect_flux3_bbox_rows(configs)
    if not rows:
        return ""
    return build_flux3_bbox_json(rows, mode)


class Flux3KleinBBox_EditUtils:
    """FLUX.3 bounding-box element table (the bbox canvas node).

    Split out of ``Flux3KleinConfigPreparer_EditUtils`` so the preparer stays a
    short, ordinary config node: this is the only place with a drawing surface,
    and it is where the taller row-editing area lives.

    Wiring (per image):

        LoadImage ─► Flux3KleinConfigPreparer ─┬─ configs ──────────┐
                                               └─ config ───────────┤
        LoadImage ─────────────────────────────── image ────────────┤
                                                  Flux3KleinBBox ───┴─► configs ─► EditTextEncode

    The node takes the chain built so far (`configs`) plus one image's `config`,
    attaches the element table to that single config, and returns the chain with
    that entry replaced - so the table reaches the model without the preparer
    having to know anything about boxes.

    If `config` is also present in `configs` (the usual case, because a preparer's
    `configs` output already contains its own `config`) the entry is REPLACED, not
    appended, so the image is never sent twice. If it is not found - e.g. `configs`
    was left unconnected - it is appended instead.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                # The chain built by the other config nodes.
                "configs": ("LIST", {"default": None, "tooltip": "Configs chain from other config nodes"}),
                # This image's single config (the preparer's `config` output).
                "config": ("ANY", {"default": None, "tooltip": "Single config for this image (the preparer's config output). The element table is attached to it."}),
                # Only used as the drawing underlay, never encoded.
                "image": ("IMAGE", {"tooltip": "Reference image shown under the boxes while drawing (guide only, not part of the output)."}),
                # No `disabled` option: this node exists only to draw the element
                # table, so connecting it should produce a table, not a no-op.
                # (To mute it temporarily, bypass the node in ComfyUI.)
                "bbox_mode": (["generation", "editing"], {
                    "default": "generation",
                    "tooltip": "generation: laying out a NEW image. Each row is "
                               "{'id','bbox','desc'} and its box says where that element goes. "
                               "editing: changing an EXISTING image. Each row is "
                               "{'id','from','src_bbox','tgt_bbox','desc'} with two boxes - where the "
                               "element is now (source) and where it should end up (target) - and names "
                               "the reference image it comes from, so one element can be kept, moved, "
                               "created or removed."}),
                "bbox_rows": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Element-table rows, owned by the canvas. Draw boxes instead of editing this by hand."}),
                # Hidden JSON mirrors. The canvas keeps these in sync and they
                # survive a workflow round-trip, so the row table is restored
                # exactly (including ids and descriptions) after reload.
                "bbox_ids": ("STRING", {"default": "", "multiline": False,
                    "tooltip": "Hidden: one element id per row (canvas-owned)."}),
                "bbox_descs": ("STRING", {"default": "", "multiline": False,
                    "tooltip": "Hidden: the row descriptions, JSON encoded (canvas-owned)."}),
            }
        }

    # `configs` already contains the decorated config grouped with its table, so
    # there is no separate `config` output to keep in sync with it.
    RETURN_TYPES = ("LIST", "STRING")
    RETURN_NAMES = ("configs", "bbox_json")
    FUNCTION = "apply_bbox"

    CATEGORY = "advanced/conditioning"

    def apply_bbox(self, configs=None, config=None, image=None,
                   bbox_mode="generation", bbox_rows="", bbox_ids="", bbox_descs=""):
        if configs is None:
            configs = []

        # Parse here so a malformed payload fails at the node the user actually
        # edited, not deep inside the encoder.
        rows, parsed_mode = [], "generation"
        if bbox_mode == "disabled":
            # `disabled` was removed from the dropdown: this node exists only to
            # build the element table, so connecting it should not silently do
            # nothing. Still accepted so an older saved workflow keeps loading.
            print("Flux3Klein bbox: bbox_mode='disabled' is no longer offered "
                  "(the node always builds a table now) - no element table will be emitted. "
                  "Set the mode to generation or editing.")
        else:
            rows, parsed_mode = parse_flux3_bbox_payload(bbox_rows, width=None, height=None)
            if rows:
                print(f"Flux3Klein bbox: {len(rows)} {parsed_mode} row(s) — "
                      f"{describe_flux3_bbox_rows(rows, parsed_mode)}")
            else:
                print(f"Flux3Klein bbox: mode={bbox_mode} but no usable rows yet "
                      "(draw a box on the canvas)")

        named_refs = sorted({
            row["from"] for row in rows
            if parsed_mode == "editing" and row.get("from")
        })

        # No single config to decorate. In generation mode that is the normal
        # FLUX.3 layout case: a text prompt plus boxes, with no reference image
        # at all - so the table still has to reach the prompt. Carry it in a
        # config-shaped, table-only entry (flagged so the encoder skips the
        # image path, since there is no image to encode).
        if config is None:
            if not rows:
                # Nothing drawn and nothing to decorate: pure pass-through.
                print("Flux3Klein bbox: no `config` connected and no rows drawn, "
                      "passing the chain through unchanged")
                return (list(configs), flux3_bbox_rows_json(configs))
            table_entry = {
                "bbox_table_only": True,
                "bbox_mode": bbox_mode,
                "bbox_rows_parsed": rows,
                "bbox_mode_parsed": parsed_mode,
                "bbox_json": build_flux3_bbox_json(rows, parsed_mode),
                "bbox_refs": named_refs,
            }
            stamp_config_id(table_entry)
            config_output = list(configs) + [table_entry]
            print("Flux3Klein bbox: no `config` connected, carrying the element table on its own "
                  f"({len(rows)} row(s)) so generation mode still sends the layout")
            return (config_output, flux3_bbox_rows_json(config_output))

        # Shallow copy: we only add/replace keys, and copying keeps the upstream
        # node's own config object unmodified.
        decorated = copy.copy(config)
        decorated["bbox_mode"] = bbox_mode
        decorated["bbox_rows_parsed"] = rows
        decorated["bbox_mode_parsed"] = parsed_mode
        decorated["bbox_json"] = build_flux3_bbox_json(rows, parsed_mode) if rows else ""
        decorated["bbox_refs"] = named_refs
        stamp_config_id(decorated)

        # Replace the matching entry rather than appending, so wiring a
        # preparer's own `configs` output here does not send the image twice.
        # Match by identity first, then by the stable config_id - a downstream
        # preparer deepcopies the chain, so a legitimate upstream entry can be a
        # copy whose identity differs but whose id matches.
        config_output = list(configs)
        replaced = False
        for index, existing in enumerate(config_output):
            if existing is config:
                config_output[index] = decorated
                replaced = True
                break
        if not replaced:
            target_id = config.get("config_id") if isinstance(config, dict) else None
            if target_id:
                for index, existing in enumerate(config_output):
                    if isinstance(existing, dict) and existing.get("config_id") == target_id:
                        config_output[index] = decorated
                        replaced = True
                        break
        if not replaced:
            config_output.append(decorated)

        # `config_output` already holds `decorated`, so no separate config output.
        return (config_output, flux3_bbox_rows_json(config_output))
