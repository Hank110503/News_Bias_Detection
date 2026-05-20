import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#111827",
        mist: "#edf2f7",
        visual: "#4a6fa5",
        textual: "#a5643f",
        multimodal: "#3a7d63",
      },
      boxShadow: {
        panel: "0 18px 60px rgba(15, 23, 42, 0.08)",
      },
      fontFamily: {
        sans: ["IBM Plex Sans", "Segoe UI Variable", "Segoe UI", "sans-serif"],
        mono: ["IBM Plex Mono", "Consolas", "monospace"],
      },
      backgroundImage: {
        "app-radial":
          "radial-gradient(circle at top left, rgba(74,111,165,0.16), transparent 34%), radial-gradient(circle at top right, rgba(165,100,63,0.14), transparent 28%), linear-gradient(180deg, #f8fafc 0%, #eef3f8 100%)",
      },
    },
  },
  plugins: [],
} satisfies Config;

