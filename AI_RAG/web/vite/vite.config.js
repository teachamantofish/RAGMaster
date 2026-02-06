import { defineConfig } from 'vite'
import FullReload from 'vite-plugin-full-reload'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const projectRoot = fileURLToPath(new URL('..', import.meta.url))
const pipelineDir = path.resolve(projectRoot, '..', 'pipeline')
const appPathsPath = path.resolve(projectRoot, 'config', 'paths.json')
const appPaths = JSON.parse(fs.readFileSync(appPathsPath, 'utf8'))
function pipelineFileProxy() {
  return {
    name: 'pipeline-file-proxy',
    apply: 'serve',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (!req.url || !req.url.startsWith('/pipeline/')) {
          next()
          return
        }

        const relativePath = req.url.replace(/^\/pipeline\//, '').split('?')[0].split('#')[0]
        if (!relativePath) {
          res.statusCode = 404
          res.end('Missing pipeline path segment.')
          return
        }

        const targetPath = path.join(pipelineDir, relativePath)
        const normalized = path.normalize(targetPath)

        if (!normalized.startsWith(pipelineDir)) {
          res.statusCode = 403
          res.end('Access to the requested resource is forbidden.')
          return
        }

        fs.readFile(normalized, (err, data) => {
          if (err) {
            res.statusCode = err.code === 'ENOENT' ? 404 : 500
            res.end(`Unable to read ${relativePath}: ${err.message}`)
            return
          }

          const ext = path.extname(normalized).toLowerCase()
          const contentType =
            ext === '.csv'
              ? 'text/csv; charset=utf-8'
              : 'text/plain; charset=utf-8'

          res.setHeader('Content-Type', contentType)
          res.end(data)
        })
      })
    },
  }
}

export default defineConfig({
  root: '..', // your source is web/
  appType: 'mpa',
  plugins: [
    FullReload(['**/*.html']), // reload on ANY html change
    pipelineFileProxy(),
  ],
  define: {
    __APP_PATHS__: JSON.stringify(appPaths),
  },
  server: {
    port: 5173,
    watch: { usePolling: true, interval: 100 }, // fixes Windows/FS watchers
    fs: { allow: [projectRoot, pipelineDir] },
  },
  build: { outDir: './build', emptyOutDir: true },
})

