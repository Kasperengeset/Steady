import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { VitePWA } from 'vite-plugin-pwa'

export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['icon.svg'],
      manifest: {
        name: 'Beyond the Clinic',
        short_name: 'Beyond',
        description: 'Daily voice check-in for maintenance after TMS treatment',
        theme_color: '#2f6f62',
        background_color: '#f4efe6',
        display: 'standalone',
        icons: [{ src: 'icon.svg', sizes: 'any', type: 'image/svg+xml', purpose: 'any' }],
      },
    }),
  ],
  server: {
    host: true,
    proxy: { '/api': 'http://localhost:8000' },
  },
})
