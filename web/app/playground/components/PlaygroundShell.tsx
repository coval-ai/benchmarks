"use client";

import type { ReactNode } from "react";

type PlaygroundShellProps = {
  /**
   * Site chrome (e.g. `PlaygroundHeader`). Pass **only** through this prop so it
   * lives in a `<header>` landmark. Do not put the same chrome inside `children`,
   * or you will duplicate markup and lose a clear banner/main split for AT.
   */
  header?: ReactNode;
  children: ReactNode;
  /** Appended to the outer surface classes (spacing, overflow, etc.). */
  className?: string;
};

/**
 * Top-level layout for `/playground`: theme surface, optional fixed header, main
 * document region. Thin on purpose — central place to add scroll lock, layout
 * variants, or providers later without touching every page.
 */
export function PlaygroundShell({ header, children, className }: PlaygroundShellProps) {
  const surfaceClassName = [
    "min-h-screen bg-surface-primary text-text-primary",
    className?.trim()
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div className={surfaceClassName}>
      {header ? <header>{header}</header> : null}
      <main>{children}</main>
    </div>
  );
}
