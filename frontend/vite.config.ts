import { fileURLToPath, URL } from 'node:url'
import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  // Dev-only proxy target. In production the frontend is served by nginx,
  // which proxies /api itself (see docker/nginx.conf).
  const backendUrl = env['VITE_BACKEND_PROXY'] ?? 'http://127.0.0.1:8000'

  return {
    plugins: [react(), tailwindcss()],
    resolve: {
      alias: {
        '@': fileURLToPath(new URL('./src', import.meta.url)),
      },
    },
    server: {
      port: 5173,
      strictPort: true,
      proxy: {
        '/api': {
          target: backendUrl,
          changeOrigin: true,
          // Server-Sent Events must not be buffered or transformed by the proxy.
          configure: (proxy) => {
            proxy.on('proxyRes', (proxyRes) => {
              const contentType = proxyRes.headers['content-type']
              if (contentType !== undefined && contentType.includes('text/event-stream')) {
                proxyRes.headers['cache-control'] = 'no-cache, no-transform'
              }
            })
          },
        },
      },
    },
    build: {
      outDir: 'dist',
      sourcemap: mode !== 'production',
    },
  }
})
