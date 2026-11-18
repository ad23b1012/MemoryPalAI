import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: [
          "-apple-system",
          "BlinkMacSystemFont",
          '"SF Pro Display"',
          '"SF Pro Text"',
          '"Plus Jakarta Sans"',
          '"Inter"',
          "system-ui",
          "-apple-system",
          "sans-serif",
        ],
        heading: [
          '"SF Pro Display"',
          "-apple-system",
          "BlinkMacSystemFont",
          '"Plus Jakarta Sans"',
          "system-ui",
          "sans-serif",
        ],
        mono: [
          '"SF Mono"',
          '"JetBrains Mono"',
          "Menlo",
          "Consolas",
          "monospace",
        ],
      },
      colors: {
        dark: {
          950: "#020617",
          900: "#060919",
          850: "#0a1024",
          800: "#0f172a",
          700: "#1e293b",
        },
        brand: {
          teal: "#14b8a6",
          cyan: "#06b6d4",
          indigo: "#6366f1",
          violet: "#8b5cf6",
          magenta: "#ec4899",
        },
      },
      animation: {
        "float": "float 6s ease-in-out infinite",
        "float-delayed": "float 6s ease-in-out 3s infinite",
        "pulse-slow": "pulse 5s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "liquid-drift-1": "drift1 18s ease-in-out infinite alternate",
        "liquid-drift-2": "drift2 24s ease-in-out infinite alternate",
        "liquid-drift-3": "drift3 20s ease-in-out infinite alternate",
        "liquid-pulse-glow": "pulseGlow 8s ease-in-out infinite alternate",
      },
      keyframes: {
        float: {
          "0%, 100%": { transform: "translateY(0px)" },
          "50%": { transform: "translateY(-8px)" },
        },
        drift1: {
          "0%": { transform: "translate(0px, 0px) scale(1)" },
          "50%": { transform: "translate(60px, -40px) scale(1.15)" },
          "100%": { transform: "translate(-40px, 30px) scale(0.92)" },
        },
        drift2: {
          "0%": { transform: "translate(0px, 0px) scale(1)" },
          "50%": { transform: "translate(-50px, 50px) scale(0.88)" },
          "100%": { transform: "translate(40px, -30px) scale(1.12)" },
        },
        drift3: {
          "0%": { transform: "translate(0px, 0px) scale(1)" },
          "50%": { transform: "translate(30px, 40px) scale(1.1)" },
          "100%": { transform: "translate(-30px, -40px) scale(0.95)" },
        },
        pulseGlow: {
          "0%": { opacity: "0.6", transform: "scale(1)" },
          "100%": { opacity: "0.95", transform: "scale(1.12)" },
        },
      },
      boxShadow: {
        "liquid-glow": "0 0 50px -5px rgba(6, 182, 212, 0.4), inset 0 1px 1px 0 rgba(255, 255, 255, 0.6)",
      },
    },
  },
  plugins: [],
};

export default config;
