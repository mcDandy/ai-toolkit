import { useRef, useState } from 'react';
import classNames from 'classnames';

// Per-box color so multiple boxes are individually distinguishable (not all one
// color). Hue is spread by the golden-angle so adjacent indices look distinct;
// stable per index so the same box keeps its color across the editor/overlay and
// matches its #n chip in the sidebar.
export function boxColor(index: number): string {
  const hue = (index * 137.508) % 360;
  return `hsl(${hue.toFixed(1)}, 80%, 60%)`;
}

// Same hue as boxColor but with an alpha, for tinting backgrounds (e.g. the
// sidebar element cards) so they visually match their box on the image.
export function boxColorAlpha(index: number, alpha: number): string {
  const hue = (index * 137.508) % 360;
  return `hsla(${hue.toFixed(1)}, 80%, 60%, ${alpha})`;
}

// A single bbox parsed from a layout caption/prompt (Ideogram or MOSS-Image).
// Stored coords are normalized 0-1000 in [y1, x1, y2, x2] order (top-left origin).
export interface OverlayBox {
  y1: number;
  x1: number;
  y2: number;
  x2: number;
  label: string;
  type: 'obj' | 'text';
}

// Same as OverlayBox but carries the element's index in the caption's element
// array, so edits can be written back to the right element.
export interface EditableBox extends OverlayBox {
  elementIndex: number;
}

// Natural pixel size of the image a caption belongs to. Needed for layouts with
// absolute-pixel boxes (MOSS-Image); normalized layouts (Ideogram) ignore it.
export interface ImageDims {
  w: number;
  h: number;
}

// Structured caption layouts with boxes. All boxes are exposed in one space:
// normalized 0-1000, [y1, x1, y2, x2], top-left origin (what the overlay/editor
// draws), and converted back to the layout's own space when edited.
//  - ideogram: {compositional_deconstruction: {elements: [{type, bbox, desc|text}]}}
//    bbox is already 0-1000 yxyx, top-left.
//  - moss:     {summary, items: [{label, caption, text?, bbox}], bbox_format, bbox_coord_space}
//    bbox_format "yxyx" (default) or "xyxy"; bbox_coord_space "abs_centralized"
//    (pixels, origin at the canvas center - the default), "abs" (pixels, top-left),
//    "norm" (0-1) or "norm_1000".
type LayoutKind = 'ideogram' | 'moss';

export function captionLayout(data: any): LayoutKind | null {
  if (!data || typeof data !== 'object') return null;
  if (Array.isArray(data?.compositional_deconstruction?.elements)) return 'ideogram';
  if (Array.isArray(data?.items) && (data.summary || data.bbox_coord_space || data.bbox_format)) return 'moss';
  return null;
}

// The element array of a layout caption (mutable, for editing), or null.
export function captionElements(data: any): any[] | null {
  const kind = captionLayout(data);
  if (kind === 'ideogram') return data.compositional_deconstruction.elements;
  if (kind === 'moss') return data.items;
  return null;
}

function num(v: any): number | null {
  const n = typeof v === 'string' ? parseFloat(v) : v;
  return typeof n === 'number' && Number.isFinite(n) ? n : null;
}

function rawBox(el: any): number[] | null {
  const bb = el?.bbox;
  if (!Array.isArray(bb) || bb.length !== 4) return null;
  const n = bb.map(num);
  return n.every(v => v !== null) ? (n as number[]) : null;
}

// [y1, x1, y2, x2] in the layout's space -> normalized 0-1000 top-left (unclamped).
function toNorm(kind: LayoutKind, data: any, bb: number[], dims?: ImageDims): BoxCoords | null {
  if (kind === 'ideogram') return { y1: bb[0], x1: bb[1], y2: bb[2], x2: bb[3] };
  const xyxy = String(data?.bbox_format ?? 'yxyx').toLowerCase() === 'xyxy';
  let [y1, x1, y2, x2] = xyxy ? [bb[1], bb[0], bb[3], bb[2]] : bb;
  const space = String(data?.bbox_coord_space ?? 'abs_centralized').toLowerCase();
  if (space.startsWith('norm')) {
    // 0-1, or 0-1000 when named so (or when the values clearly are)
    const scale = space.includes('1000') || Math.max(Math.abs(y2), Math.abs(x2)) > 1.5 ? 1 : 1000;
    return { y1: y1 * scale, x1: x1 * scale, y2: y2 * scale, x2: x2 * scale };
  }
  if (!dims || dims.w <= 0 || dims.h <= 0) return null;
  if (space.includes('central') || space.includes('center')) {
    y1 += dims.h / 2;
    y2 += dims.h / 2;
    x1 += dims.w / 2;
    x2 += dims.w / 2;
  }
  return { y1: (y1 / dims.h) * 1000, x1: (x1 / dims.w) * 1000, y2: (y2 / dims.h) * 1000, x2: (x2 / dims.w) * 1000 };
}

// Normalized 0-1000 top-left box -> the layout's own bbox array.
function fromNorm(kind: LayoutKind, data: any, b: BoxCoords, dims?: ImageDims): number[] | null {
  if (kind === 'ideogram') return [b.y1, b.x1, b.y2, b.x2];
  const space = String(data?.bbox_coord_space ?? 'abs_centralized').toLowerCase();
  let out: number[];
  if (space.startsWith('norm')) {
    const s = space.includes('1000') ? 1 : 1 / 1000;
    out = [b.y1 * s, b.x1 * s, b.y2 * s, b.x2 * s].map(v => Math.round(v * 10000) / 10000);
  } else {
    if (!dims || dims.w <= 0 || dims.h <= 0) return null;
    const cy = space.includes('central') || space.includes('center') ? dims.h / 2 : 0;
    const cx = space.includes('central') || space.includes('center') ? dims.w / 2 : 0;
    out = [
      Math.round((b.y1 / 1000) * dims.h - cy),
      Math.round((b.x1 / 1000) * dims.w - cx),
      Math.round((b.y2 / 1000) * dims.h - cy),
      Math.round((b.x2 / 1000) * dims.w - cx),
    ];
  }
  const xyxy = String(data?.bbox_format ?? 'yxyx').toLowerCase() === 'xyxy';
  return xyxy ? [out[1], out[0], out[3], out[2]] : out;
}

function elementLabel(kind: LayoutKind, el: any): { label: string; type: 'obj' | 'text' } {
  const isText = kind === 'ideogram' ? el?.type === 'text' : el?.label === 'text' || typeof el?.text === 'string';
  const label = isText ? (el?.text ?? '') : kind === 'ideogram' ? (el?.desc ?? '') : (el?.caption ?? el?.label ?? '');
  return { label: `${label}`, type: isText ? 'text' : 'obj' };
}

// Write an edited (normalized) box back into an element, in the layout's space.
export function setElementBox(data: any, el: any, box: BoxCoords, dims?: ImageDims): boolean {
  const kind = captionLayout(data);
  if (!kind || !el) return false;
  const bb = fromNorm(kind, data, box, dims);
  if (!bb) return false;
  el.bbox = bb;
  return true;
}

// A new element of the caption's layout for a box drawn in the editor.
export function newBoxElement(data: any, box: BoxCoords, dims?: ImageDims): any | null {
  const kind = captionLayout(data);
  if (!kind) return null;
  const bb = fromNorm(kind, data, box, dims);
  if (!bb) return null;
  return kind === 'ideogram' ? { type: 'obj', bbox: bb, desc: '' } : { label: 'object', caption: '', bbox: bb };
}

// True for a JSON caption whose layout has boxes we can show (either kind).
export function isLayoutCaption(text: string): boolean {
  const t = text.trim();
  if (!t.startsWith('{')) return false;
  try {
    return captionLayout(JSON.parse(t)) !== null;
  } catch {
    return false;
  }
}

// Returns the list of boxes if the text is a layout caption/prompt (Ideogram or
// MOSS-Image) with at least one usable bbox, otherwise null (normal captions get
// no overlay). Absolute-pixel layouts need `dims`.
export function parseBoundingBoxes(text: string, dims?: ImageDims): OverlayBox[] | null {
  const trimmed = text.trim();
  if (!trimmed.startsWith('{')) return null;
  let data: any;
  try {
    data = JSON.parse(trimmed);
  } catch {
    return null;
  }
  const boxes = extractBoxes(data, dims);
  return boxes.length > 0 ? boxes.map(({ elementIndex: _i, ...b }) => b) : null;
}

// Build editable boxes (tagged with their element index) from an already-parsed
// caption object. Returns [] if the object has no usable elements/boxes.
export function extractBoxes(data: any, dims?: ImageDims): EditableBox[] {
  const kind = captionLayout(data);
  const elements = captionElements(data);
  if (!kind || !elements) return [];
  const boxes: EditableBox[] = [];
  elements.forEach((el: any, idx: number) => {
    const bb = rawBox(el);
    if (!bb) return;
    const n = toNorm(kind, data, bb, dims);
    if (!n) return;
    boxes.push({ ...normalizeBox(n), ...elementLabel(kind, el), elementIndex: idx });
  });
  return boxes;
}

// Parse the caption for EDITING: returns the parsed object plus its boxes. null if
// the text is not a layout caption with at least one box.
export function parseCaptionForEditing(text: string, dims?: ImageDims): { data: any; boxes: EditableBox[] } | null {
  const trimmed = text.trim();
  if (!trimmed.startsWith('{')) return null;
  let data: any;
  try {
    data = JSON.parse(trimmed);
  } catch {
    return null;
  }
  const boxes = extractBoxes(data, dims);
  return boxes.length > 0 ? { data, boxes } : null;
}

export interface BoxCoords {
  y1: number;
  x1: number;
  y2: number;
  x2: number;
}

// Clamp/round/normalize coords to integer 0-1000 with y1<y2, x1<x2.
function normalizeBox(b: { x1: number; y1: number; x2: number; y2: number }): BoxCoords {
  const cl = (v: number) => Math.max(0, Math.min(1000, Math.round(v)));
  const x1 = cl(Math.min(b.x1, b.x2));
  const x2 = cl(Math.max(b.x1, b.x2));
  const y1 = cl(Math.min(b.y1, b.y2));
  const y2 = cl(Math.max(b.y1, b.y2));
  return { y1, x1, y2, x2 };
}

type DragMode = 'move' | 'nw' | 'ne' | 'sw' | 'se';
const MIN_SIZE = 8; // minimum normalized box span so boxes can't collapse

// Editable overlay: click to select, drag a box body to move, drag a corner
// handle to resize, ✕ to delete. With `drawing` on, drag on empty space to
// create a new box. All geometry commits fire on pointer-up. Place inside the
// same `relative` image-wrapping container as BoundingBoxOverlay; coordinate
// math uses this layer's own rect, so it's correct at any zoom.
export function BoundingBoxEditor({
  boxes,
  selectedIndex,
  drawing,
  onSelect,
  onChangeBox,
  onCreateBox,
}: {
  boxes: EditableBox[];
  selectedIndex: number | null;
  drawing: boolean;
  onSelect: (elementIndex: number | null) => void;
  onChangeBox: (elementIndex: number, box: BoxCoords) => void;
  onCreateBox: (box: BoxCoords) => void;
}) {
  const rootRef = useRef<HTMLDivElement>(null);
  const [draft, setDraft] = useState<{ index: number; box: OverlayBox } | null>(null);
  const [drawDraft, setDrawDraft] = useState<{ x1: number; y1: number; x2: number; y2: number } | null>(null);

  const toNorm = (clientX: number, clientY: number) => {
    const rect = rootRef.current!.getBoundingClientRect();
    return {
      x: Math.max(0, Math.min(1000, ((clientX - rect.left) / rect.width) * 1000)),
      y: Math.max(0, Math.min(1000, ((clientY - rect.top) / rect.height) * 1000)),
    };
  };

  const startDrag = (e: React.PointerEvent, box: EditableBox, mode: DragMode) => {
    // Stop the event from reaching the zoom/pan wrapper so dragging never pans.
    e.preventDefault();
    e.stopPropagation();
    onSelect(box.elementIndex);
    const root = rootRef.current;
    if (!root) return;
    const startX = e.clientX;
    const startY = e.clientY;
    const start: OverlayBox = { ...box };
    let current: OverlayBox = { ...box };

    const move = (ev: PointerEvent) => {
      const rect = root.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      const dx = ((ev.clientX - startX) / rect.width) * 1000;
      const dy = ((ev.clientY - startY) / rect.height) * 1000;
      let { x1, y1, x2, y2 } = start;
      if (mode === 'move') {
        const w = x2 - x1;
        const h = y2 - y1;
        const nx1 = Math.max(0, Math.min(1000 - w, x1 + dx));
        const ny1 = Math.max(0, Math.min(1000 - h, y1 + dy));
        x1 = nx1;
        y1 = ny1;
        x2 = nx1 + w;
        y2 = ny1 + h;
      } else {
        if (mode.includes('w')) x1 = Math.max(0, Math.min(x2 - MIN_SIZE, x1 + dx));
        if (mode.includes('e')) x2 = Math.min(1000, Math.max(x1 + MIN_SIZE, x2 + dx));
        if (mode.includes('n')) y1 = Math.max(0, Math.min(y2 - MIN_SIZE, y1 + dy));
        if (mode.includes('s')) y2 = Math.min(1000, Math.max(y1 + MIN_SIZE, y2 + dy));
      }
      current = { ...start, x1, y1, x2, y2 };
      setDraft({ index: box.elementIndex, box: current });
    };

    const up = () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
      setDraft(null);
      onChangeBox(box.elementIndex, normalizeBox(current));
    };

    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
  };

  // Pointer-down on the empty root: draw a new box (drawing mode) or deselect.
  const onRootPointerDown = (e: React.PointerEvent) => {
    if (!drawing) {
      onSelect(null);
      return;
    }
    e.preventDefault();
    e.stopPropagation();
    if (!rootRef.current) return;
    const p0 = toNorm(e.clientX, e.clientY);
    let cur = { x1: p0.x, y1: p0.y, x2: p0.x, y2: p0.y };
    setDrawDraft(cur);

    const move = (ev: PointerEvent) => {
      const p = toNorm(ev.clientX, ev.clientY);
      cur = { x1: p0.x, y1: p0.y, x2: p.x, y2: p.y };
      setDrawDraft({ ...cur });
    };
    const up = () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
      setDrawDraft(null);
      const nb = normalizeBox(cur);
      if (nb.x2 - nb.x1 >= MIN_SIZE && nb.y2 - nb.y1 >= MIN_SIZE) onCreateBox(nb);
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
  };

  const handles: { mode: DragMode; cls: string }[] = [
    { mode: 'nw', cls: 'top-0 left-0 -translate-x-1/2 -translate-y-1/2 cursor-nwse-resize' },
    { mode: 'ne', cls: 'top-0 right-0 translate-x-1/2 -translate-y-1/2 cursor-nesw-resize' },
    { mode: 'sw', cls: 'bottom-0 left-0 -translate-x-1/2 translate-y-1/2 cursor-nesw-resize' },
    { mode: 'se', cls: 'bottom-0 right-0 translate-x-1/2 translate-y-1/2 cursor-nwse-resize' },
  ];

  return (
    <div
      ref={rootRef}
      onPointerDown={onRootPointerDown}
      className={classNames('absolute inset-0', drawing ? 'cursor-crosshair' : '')}
    >
      {boxes.map(box => {
        const pos = draft && draft.index === box.elementIndex ? draft.box : box;
        const selected = selectedIndex === box.elementIndex;
        const showSelected = selected && !drawing;
        const color = boxColor(box.elementIndex);
        return (
          <div
            key={box.elementIndex}
            onPointerDown={drawing ? undefined : e => startDrag(e, box, 'move')}
            className={classNames('absolute border-2', {
              'pointer-events-none': drawing,
              'cursor-move touch-none': !drawing,
              'ring-2 ring-blue-400': showSelected,
            })}
            style={{
              left: `${pos.x1 / 10}%`,
              top: `${pos.y1 / 10}%`,
              width: `${(pos.x2 - pos.x1) / 10}%`,
              height: `${(pos.y2 - pos.y1) / 10}%`,
              borderColor: showSelected ? '#ffffff' : color,
            }}
          >
            {box.label && (
              <span
                title={box.label}
                className="absolute top-0 left-0 max-w-full px-1 py-0.5 text-[9px] leading-tight font-medium whitespace-pre-line break-words line-clamp-2 text-gray-900 pointer-events-none"
                style={{ backgroundColor: color }}
              >
                {box.label}
              </span>
            )}
            {selected &&
              !drawing &&
              handles.map(h => (
                <div
                  key={h.mode}
                  onPointerDown={e => startDrag(e, { ...box, ...pos }, h.mode)}
                  className={classNames(
                    'absolute w-3 h-3 rounded-sm border border-gray-900 bg-white touch-none',
                    h.cls,
                  )}
                />
              ))}
          </div>
        );
      })}
      {drawDraft && (
        <div
          className="absolute border-2 border-dashed border-white bg-white/10 pointer-events-none"
          style={{
            left: `${Math.min(drawDraft.x1, drawDraft.x2) / 10}%`,
            top: `${Math.min(drawDraft.y1, drawDraft.y2) / 10}%`,
            width: `${Math.abs(drawDraft.x2 - drawDraft.x1) / 10}%`,
            height: `${Math.abs(drawDraft.y2 - drawDraft.y1) / 10}%`,
          }}
        />
      )}
    </div>
  );
}

// Absolute overlay layer of boxes. Place inside a `relative` container that wraps
// the image (e.g. inside react-zoom-pan-pinch's TransformComponent) so boxes
// track the image during zoom/pan. Coords are percentages of the frame, so this
// is resolution-independent. pointer-events-none keeps zoom/pan working through it.
export default function BoundingBoxOverlay({ boxes }: { boxes: OverlayBox[] }) {
  return (
    <div className="absolute inset-0 pointer-events-none">
      {boxes.map((b, i) => {
        const color = boxColor(i);
        return (
          <div
            key={i}
            className="absolute border-2"
            style={{
              left: `${b.x1 / 10}%`,
              top: `${b.y1 / 10}%`,
              width: `${(b.x2 - b.x1) / 10}%`,
              height: `${(b.y2 - b.y1) / 10}%`,
              borderColor: color,
            }}
          >
            {b.label && (
              <span
                title={b.label}
                className="absolute top-0 left-0 max-w-full px-1 py-0.5 text-[9px] leading-tight font-medium whitespace-pre-line break-words line-clamp-3 text-gray-900"
                style={{ backgroundColor: color }}
              >
                {b.label}
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
}
