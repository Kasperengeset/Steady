import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { VitePWA } from 'vite-plugin-pwa'

export default defineConfig(({ mode }) => ({
  plugins: [
    react(),
    tailwindcss(),
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['icon.svg'],
      devOptions: { enabled: true, type: 'classic', navigateFallback: 'index.html' },
      workbox: { importScripts: ['/push-sw.js'], navigateFallbackDenylist: [/^\/api\//] },
      manifest: {
        name: 'Steady',
        short_name: 'Steady',
        description: 'Staying well at home after TMS treatment',
        theme_color: '#43654f',
        background_color: '#f9faf6',
        display: 'standalone',
        icons: [{ src: 'icon.svg', sizes: 'any', type: 'image/svg+xml', purpose: 'any' }],
      },
    }),
  ],
  server: {
    host: true,
    proxy: { '/api': loadEnv(mode, '.', 'STEADY_').STEADY_API_TARGET || 'http://127.0.0.1:8000' },
  },
}))
