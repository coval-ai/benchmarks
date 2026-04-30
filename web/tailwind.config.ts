import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./hooks/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Surface colors
        "surface-primary": "var(--color-surface-primary)",
        "surface-secondary": "var(--color-surface-secondary)",
        "surface-elevated": "var(--color-surface-elevated)",
        "surface-tooltip": "var(--color-surface-tooltip)",
        "surface-overlay": "var(--color-surface-overlay)",
        // Text colors
        "text-primary": "var(--color-text-primary)",
        "text-secondary": "var(--color-text-secondary)",
        "text-tertiary": "var(--color-text-tertiary)",
        // Border colors
        "border-primary": "var(--color-border-primary)",
        "border-secondary": "var(--color-border-secondary)",
        // Interactive
        "hover-bg": "var(--color-hover-bg)",
        "selected-bg": "var(--color-selected-bg)",
        "selected-border": "var(--color-selected-border)",
        // Toggle
        "surface-toggle-active": "var(--color-surface-toggle-active)",
        "text-on-toggle-active": "var(--color-text-on-toggle-active)",
        "surface-toggle-inactive": "var(--color-surface-toggle-inactive)",
        "text-on-toggle-inactive": "var(--color-text-on-toggle-inactive)",
        // Spinner
        "spinner-track": "var(--color-spinner-track)",
        "spinner-head": "var(--color-spinner-head)",
        // Chart colors
        "chart-grid": "var(--color-chart-grid)",
        "chart-axis-text": "var(--color-chart-axis-text)",
        "chart-label": "var(--color-chart-label)",
        "chart-median": "var(--color-chart-median)",
        "chart-box-fill": "var(--color-chart-box-fill)",
        "chart-bar-stroke": "var(--color-chart-bar-stroke)",
      },
      fontFamily: {
        sans: ["var(--font-montserrat)", "Montserrat", "Arial", "Helvetica", "sans-serif"],
        mono: ["var(--font-montserrat)", "Montserrat", "Arial", "Helvetica", "sans-serif"],
      },
    },
  },
  plugins: [],
};

export default config;
