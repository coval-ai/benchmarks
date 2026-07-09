"use client";

import type { ProviderVisual } from "@/lib/playground/provider-styles";

type ModelPillProps = {
  label: string;
  providerLabel?: string;
  visual: ProviderVisual;
  selected?: boolean;
  disabled?: boolean;
  onToggle?: () => void;
  fullWidth?: boolean;
};

export function ModelPill({
  label,
  providerLabel,
  visual,
  selected = true,
  disabled = false,
  onToggle,
  fullWidth = false
}: ModelPillProps) {
  const showProvider =
    providerLabel && !label.toLowerCase().includes(providerLabel.toLowerCase());

  const className = [
    fullWidth
      ? "flex w-full max-w-full min-w-0 items-center justify-between gap-2 rounded-xl border px-3 py-2 text-xs font-medium transition-colors"
      : "inline-flex max-w-full min-w-0 items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium transition-colors",
    selected ? "bg-surface-primary text-text-primary" : "bg-transparent text-text-tertiary",
    disabled ? "cursor-not-allowed opacity-50" : ""
  ].join(" ");

  const dotStyle = selected
    ? { backgroundColor: visual.dot }
    : { backgroundColor: "rgb(82 82 91)" };

  const borderStyle = selected
    ? { borderColor: visual.border }
    : { borderColor: "var(--color-border-primary, rgba(0, 0, 0, 0.1))" };

  const content = fullWidth ? (
    <>
      <span className="flex min-w-0 items-center gap-2">
        <span className="size-1.5 shrink-0 rounded-full" style={dotStyle} aria-hidden />
        <span className="min-w-0 truncate">{label}</span>
      </span>
      {showProvider ? (
        <span className="shrink-0 text-[10px] font-medium uppercase tracking-wide text-text-tertiary dark:text-zinc-500">
          {providerLabel}
        </span>
      ) : null}
    </>
  ) : (
    <>
      <span className="size-1.5 shrink-0 rounded-full" style={dotStyle} aria-hidden />
      <span className="min-w-0 truncate">
        {label}
        {showProvider ? (
          <span className="font-normal text-text-tertiary dark:text-zinc-500"> · {providerLabel}</span>
        ) : null}
      </span>
    </>
  );

  if (onToggle) {
    return (
      <button
        type="button"
        onClick={onToggle}
        aria-pressed={selected}
        disabled={disabled}
        className={className}
        style={borderStyle}
      >
        {content}
      </button>
    );
  }

  return (
    <span className={className} style={borderStyle}>
      {content}
    </span>
  );
}
