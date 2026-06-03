import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// During `npm run dev` the app is served on :5173 and calls `/api/...`; Vite proxies those to the
// FastAPI backend on :8000 (so there is no cross-origin hop in dev). In production the built
// `dist/` is served by uvicorn itself, so `/api` is same-origin there too.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8000',
    },
  },
})
