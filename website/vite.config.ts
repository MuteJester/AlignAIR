import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Deployed to GitHub Pages at https://mutejester.github.io/AlignAIR/ , so assets live under /AlignAIR/.
// Override with BASE_PATH at build time (e.g. "/" for a custom domain).
const base = process.env.BASE_PATH ?? "/AlignAIR/";

export default defineConfig({
  base,
  plugins: [react()],
  build: {
    outDir: "dist",
    sourcemap: false,
  },
});
