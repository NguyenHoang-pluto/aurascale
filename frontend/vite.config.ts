import { fileURLToPath, URL } from 'node:url'
import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  // Both this directory and the repository root, because the documented
  // configuration file is the root `.env` (it also carries the backend's own
  // HOST/PORT) while Vite runs from `frontend/` and would otherwise never read
  // it. A `frontend/.env` still wins, so a local override stays possible.
  const rootDir = fileURLToPath(new URL('..', import.meta.url))
  const env = { ...loadEnv(mode, rootDir, ''), ...loadEnv(mode, process.cwd(), '') }

  // Dev-only proxy target. In production the frontend is served by nginx,
  // which proxies /api itself (see docker/nginx.conf).
  const backendUrl = env['VITE_BACKEND_PROXY'] ?? 'http://127.0.0.1:8000'

  // Dev-only. Extra Host headers this dev server will answer for.
  //
  // Vite rejects requests whose Host it does not recognise, which is DNS
  // rebinding protection and worth keeping. A tunnel breaks that assumption
  // honestly: the browser asks for a random `*.trycloudflare.com` name, the
  // tunnel forwards it to 127.0.0.1:5173 with that Host intact, and Vite
  // refuses it. Naming the suffix here is what lets a temporary public test
  // work without weakening the rule for anyone who does not opt in.
  //
  // Comma-separated, and **empty by default**: a clean checkout keeps Vite's
  // stock localhost-only behaviour, and only a local `.env` turns this on. A
  // leading dot matches subdomains, so `.trycloudflare.com` is the whole
  // quick-tunnel family and nothing else.
  const allowedHosts = (env['VITE_ALLOWED_HOSTS'] ?? '')
    .split(',')
    .map((host) => host.trim())
    .filter((host) => host.length > 0)

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
      // Spread rather than set: an empty list must leave Vite's own default in
      // place, not replace it with a list that allows nothing.
      ...(allowedHosts.length > 0 ? { allowedHosts } : {}),
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
