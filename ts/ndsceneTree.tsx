import React, { useMemo, useState } from "react";

const MAX_CHILDREN = 24;

export function describeValueKind(value: unknown): string {
  if (value === null) {
    return "null";
  }
  if (Array.isArray(value)) {
    return "array";
  }
  return typeof value;
}

export function summarizeShapeValue(value: unknown): string {
  if (Array.isArray(value)) {
    return `[${value.map((entry) => summarizeShapeValue(entry)).join(", ")}]`;
  }
  if (typeof value === "number") {
    return `${value}`;
  }
  if (!value || typeof value !== "object") {
    return String(value);
  }

  const record = value as Record<string, unknown>;
  const name = typeof record.name === "string" ? record.name : typeof record.key === "string" ? record.key : "";
  const size = record.size !== undefined ? `x${String(record.size)}` : "";
  const dtype = typeof record.dtype === "string" ? `<${record.dtype}>` : "";
  const childShape = Array.isArray(record.shape) && record.shape.length > 0
    ? summarizeShapeValue(record.shape)
    : "";
  const dataTag = record.data !== undefined ? " =data" : "";

  return `${name}${size}${dtype}${childShape}${dataTag}`.trim() || "{}";
}

export function summarizeStructuredValue(label: string, value: unknown): string {
  if (label === "shape") {
    return summarizeShapeValue(value);
  }
  if (Array.isArray(value)) {
    return `${value.length} item${value.length === 1 ? "" : "s"}`;
  }
  if (value && typeof value === "object") {
    const record = value as Record<string, unknown>;
    const keys = Object.keys(record);
    let base = "";
    let hasShape = false;

    if (typeof record.name === "string") {
      base += ` "${record.name}" `;
    } else if (typeof record.key === "string") {
      base += ` "${record.key}" `;
    }
    if ("shape" in record) {
      base += ` ${summarizeShapeValue(record.shape)}`;
      hasShape = true;
    }
    if (typeof record.dtype === "string") {
      base += ` <${record.dtype}>`;
    }
    if ("data" in record) {
      base += " (data)";
    }
    if (Array.isArray(record.children)) {
      base += ` {${record.children.length}c}`;
      hasShape = true;
    }
    if ("pose" in record) {
      base += " *pose";
      hasShape = true;
    }
    if ("unpose" in record) {
      base += " *unpose";
      hasShape = true;
    }
    if (!hasShape) {
      base += keys.length === 0 ? " { }" : ` {.${keys.length}}`;
    }
    return base;
  }
  return String(value);
}

function formatLeafValue(value: unknown): string {
  if (typeof value === "string") {
    return `"${value}"`;
  }
  if (value === null) {
    return "null";
  }
  return String(value);
}

function entryList(value: unknown): Array<[string, unknown]> {
  if (Array.isArray(value)) {
    const entries = value.slice(0, MAX_CHILDREN).map((entry, index) => [String(index), entry] as [string, unknown]);
    if (value.length > MAX_CHILDREN) {
      entries.push([`…`, `${value.length - MAX_CHILDREN} more items`]);
    }
    return entries;
  }
  if (value && typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>).slice(0, MAX_CHILDREN);
    const keys = Object.keys(value as Record<string, unknown>);
    if (keys.length > MAX_CHILDREN) {
      entries.push([`…`, `${keys.length - MAX_CHILDREN} more fields`]);
    }
    return entries;
  }
  return [];
}

function NdSceneTreeNode({ label, value, depth = 0 }: { label: string; value: unknown; depth?: number }) {
  const kind = describeValueKind(value);
  const entries = useMemo(() => entryList(value), [value]);
  const expandable = entries.length > 0;
  const [expanded, setExpanded] = useState(depth < 2);
  const summary = expandable ? summarizeStructuredValue(label, value) : "";

  return (
    <div className="ndscene-node">
      <div className="ndscene-row">
        <button
          type="button"
          className="ndscene-toggle"
          data-empty={expandable ? "false" : "true"}
          onClick={() => setExpanded((current) => !current)}
        >
          {expandable ? (expanded ? "−" : "+") : "·"}
        </button>
        <div className="ndscene-key">{label}</div>
        <div className="ndscene-kind">{kind}</div>
        {expandable ? (
          <div className="ndscene-summary">{summary}</div>
        ) : (
          <div className="ndscene-value" data-value-type={kind}>
            {formatLeafValue(value)}
          </div>
        )}
      </div>
      {expandable && expanded ? (
        <div className="ndscene-children">
          {entries.map(([childLabel, childValue]) => (
            <NdSceneTreeNode
              key={`${label}.${childLabel}`}
              label={childLabel}
              value={childValue}
              depth={depth + 1}
            />
          ))}
        </div>
      ) : null}
    </div>
  );
}

export function NdSceneTree({ value, rootLabel = "scene" }: { value: unknown; rootLabel?: string }) {
  return <NdSceneTreeNode label={rootLabel} value={value} depth={0} />;
}
