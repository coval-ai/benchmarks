"use client";

import type { ReactNode } from "react";

type PlaygroundShellProps = {
  /**
   * Site chrome (e.g. `DashboardHeader`), which supplies its own `<header>`
   * banner landmark. Pass **only** through this prop so it stays a sibling of
   * `<main>`; do not also place the same chrome inside `children`.
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
      {header}
      <main>{children}</main>
    </div>
  );
}
