import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/health": "http://127.0.0.1:8765",
      "/readiness": "http://127.0.0.1:8765",
      "/models": "http://127.0.0.1:8765",
      "/registry": "http://127.0.0.1:8765",
      "/mock": "http://127.0.0.1:8765",
    },
  },
});
