import { defineConfig } from 'vite'
import FullReload from 'vite-plugin-full-reload'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'
import { spawn } from 'child_process'

const projectRoot = fileURLToPath(new URL('..', import.meta.url))
const repoRoot = path.resolve(projectRoot, '..')
const pipelineDir = path.resolve(projectRoot, '..', 'pipeline')
const getAndChunkDir = path.resolve(projectRoot, '..', 'Get_and_Chunk')
const vectorDBDir = path.resolve(projectRoot, '..', 'VectorDB')
const appPathsPath = path.resolve(projectRoot, 'config', 'paths.json')
const appPaths = JSON.parse(fs.readFileSync(appPathsPath, 'utf8'))

const staticBuildEntries = [
  'home.html',
  'source.html',
  'crawl.html',
  'crawl_pdf.html',
  'clean.html',
  'chunk.html',
  'summary.html',
  'training_data.html',
  'embed.html',
  'vector.html',
  'style.css',
  'js',
  'css',
  'images',
  'config',
]

function copyPathIfExists(fromPath, toPath) {
  if (!fs.existsSync(fromPath)) {
    return
  }

  const stats = fs.statSync(fromPath)
  if (stats.isDirectory()) {
    fs.mkdirSync(toPath, { recursive: true })
    const children = fs.readdirSync(fromPath)
    for (const child of children) {
      copyPathIfExists(path.join(fromPath, child), path.join(toPath, child))
    }
    return
  }

  fs.mkdirSync(path.dirname(toPath), { recursive: true })
  fs.copyFileSync(fromPath, toPath)
}

function copyStaticBuildResources() {
  return {
    name: 'copy-static-build-resources',
    apply: 'build',
    closeBundle() {
      const outDir = path.resolve(projectRoot, 'build')
      for (const entry of staticBuildEntries) {
        const sourcePath = path.resolve(projectRoot, entry)
        const targetPath = path.resolve(outDir, entry)
        copyPathIfExists(sourcePath, targetPath)
      }
    },
  }
}

/**
 * Generic file proxy factory for serving files from a mapped directory.
 * @param {string} urlPrefix - URL path prefix (e.g. '/pipeline/' or '/Get_and_Chunk/').
 * @param {string} baseDir - Absolute directory to serve from.
 * @param {string} pluginName - Vite plugin name.
 */
function createFileProxy(urlPrefix, baseDir, pluginName) {
  return {
    name: pluginName,
    apply: 'serve',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        // Only handle GET/HEAD — let PUT/POST fall through to write proxy
        if (!req.url || !req.url.startsWith(urlPrefix) || (req.method !== 'GET' && req.method !== 'HEAD')) {
          next()
          return
        }

        const regex = new RegExp(`^${urlPrefix.replace(/\//g, '\\/')}`)
        const relativePath = req.url.replace(regex, '').split('?')[0].split('#')[0]
        if (!relativePath) {
          res.statusCode = 404
          res.end(`Missing path segment for ${urlPrefix}.`)
          return
        }

        const targetPath = path.join(baseDir, relativePath)
        const normalized = path.normalize(targetPath)

        if (!normalized.startsWith(baseDir)) {
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
          const mimeMap = {
            '.csv': 'text/csv; charset=utf-8',
            '.json': 'application/json; charset=utf-8',
            '.py': 'text/plain; charset=utf-8',
          }
          res.setHeader('Content-Type', mimeMap[ext] || 'text/plain; charset=utf-8')
          res.end(data)
        })
      })
    },
  }
}

/**
 * Write-capable proxy: accepts PUT requests to save file content back to disk.
 * Only allows writes to mapped directories (Get_and_Chunk, pipeline) during dev.
 */
function createWriteProxy(urlPrefix, baseDir, pluginName) {
  return {
    name: pluginName,
    apply: 'serve',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (!req.url || !req.url.startsWith(urlPrefix) || req.method !== 'PUT') {
          next()
          return
        }

        const regex = new RegExp(`^${urlPrefix.replace(/\//g, '\\/')}`)
        const relativePath = req.url.replace(regex, '').split('?')[0].split('#')[0]
        if (!relativePath) {
          res.statusCode = 400
          res.end(JSON.stringify({ success: false, error: 'Missing path segment.' }))
          return
        }

        const targetPath = path.join(baseDir, relativePath)
        const normalized = path.normalize(targetPath)

        if (!normalized.startsWith(baseDir)) {
          res.statusCode = 403
          res.end(JSON.stringify({ success: false, error: 'Forbidden.' }))
          return
        }

        // Only allow writing to .py and .json config files
        const ext = path.extname(normalized).toLowerCase()
        if (ext !== '.py' && ext !== '.json') {
          res.statusCode = 403
          res.end(JSON.stringify({ success: false, error: `Writing ${ext} files is not allowed.` }))
          return
        }

        let body = ''
        req.on('data', chunk => { body += chunk })
        req.on('end', () => {
          fs.writeFile(normalized, body, 'utf8', (err) => {
            res.setHeader('Content-Type', 'application/json; charset=utf-8')
            if (err) {
              res.statusCode = 500
              res.end(JSON.stringify({ success: false, error: err.message }))
            } else {
              res.statusCode = 200
              res.end(JSON.stringify({ success: true }))
            }
          })
        })
      })
    },
  }
}

function pipelineFileProxy() {
  return createFileProxy('/pipeline/', pipelineDir, 'pipeline-file-proxy')
}

function getAndChunkFileProxy() {
  return createFileProxy('/Get_and_Chunk/', getAndChunkDir, 'get-and-chunk-file-proxy')
}

function getAndChunkWriteProxy() {
  return createWriteProxy('/Get_and_Chunk/', getAndChunkDir, 'get-and-chunk-write-proxy')
}

function vectorDBFileProxy() {
  return createFileProxy('/VectorDB/', vectorDBDir, 'vectordb-file-proxy')
}

function vectorDBWriteProxy() {
  return createWriteProxy('/VectorDB/', vectorDBDir, 'vectordb-write-proxy')
}

/**
 * Run-script API: POST /api/run-script
 * Body: { "script": "Get_and_Chunk/3.00chunker.py" }
 * Spawns Python subprocess and streams combined stdout+stderr back as JSON.
 * Only available during dev (apply: 'serve').
 */
function createRunScriptApi() {
  // Allowed script prefixes (relative to repo root)
  const allowedPrefixes = ['Get_and_Chunk/', 'VectorDB/']

  return {
    name: 'run-script-api',
    apply: 'serve',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (req.url !== '/api/run-script' || req.method !== 'POST') {
          next()
          return
        }

        let body = ''
        req.on('data', chunk => { body += chunk })
        req.on('end', () => {
          let payload
          try {
            payload = JSON.parse(body)
          } catch {
            res.statusCode = 400
            res.setHeader('Content-Type', 'application/json')
            res.end(JSON.stringify({ success: false, error: 'Invalid JSON body.' }))
            return
          }

          const script = payload.script
          if (!script || typeof script !== 'string') {
            res.statusCode = 400
            res.setHeader('Content-Type', 'application/json')
            res.end(JSON.stringify({ success: false, error: 'Missing "script" field.' }))
            return
          }

          // Security: only allow scripts under known prefixes
          if (!allowedPrefixes.some(p => script.startsWith(p))) {
            res.statusCode = 403
            res.setHeader('Content-Type', 'application/json')
            res.end(JSON.stringify({ success: false, error: `Script path not allowed: ${script}` }))
            return
          }

          const scriptPath = path.resolve(repoRoot, script)
          const normalized = path.normalize(scriptPath)
          if (!normalized.startsWith(repoRoot)) {
            res.statusCode = 403
            res.setHeader('Content-Type', 'application/json')
            res.end(JSON.stringify({ success: false, error: 'Path traversal not allowed.' }))
            return
          }

          if (!fs.existsSync(normalized)) {
            res.statusCode = 404
            res.setHeader('Content-Type', 'application/json')
            res.end(JSON.stringify({ success: false, error: `Script not found: ${script}` }))
            return
          }

          // Spawn Python process
          const pythonCmd = process.platform === 'win32' ? 'python' : 'python3'
          const child = spawn(pythonCmd, [normalized], {
            cwd: repoRoot,
            env: { ...process.env },
            stdio: ['ignore', 'pipe', 'pipe'],
          })

          let output = ''
          child.stdout.on('data', d => { output += d.toString() })
          child.stderr.on('data', d => { output += d.toString() })

          child.on('error', err => {
            res.statusCode = 500
            res.setHeader('Content-Type', 'application/json')
            res.end(JSON.stringify({ success: false, error: `Failed to spawn: ${err.message}` }))
          })

          child.on('close', code => {
            res.setHeader('Content-Type', 'application/json')
            res.statusCode = code === 0 ? 200 : 500
            res.end(JSON.stringify({
              success: code === 0,
              exitCode: code,
              output: output,
            }))
          })
        })
      })
    },
  }
}

export default defineConfig({
  root: '..', // your source is web/
  base: './',
  appType: 'mpa',
  plugins: [
    FullReload(['**/*.html']), // reload on ANY html change
    pipelineFileProxy(),
    getAndChunkFileProxy(),
    getAndChunkWriteProxy(),
    vectorDBFileProxy(),
    vectorDBWriteProxy(),
    createRunScriptApi(),
    copyStaticBuildResources(),
  ],
  define: {
    __APP_PATHS__: JSON.stringify(appPaths),
  },
  server: {
    port: 5173,
    watch: { usePolling: true, interval: 100 }, // fixes Windows/FS watchers
    fs: { allow: [projectRoot, pipelineDir, getAndChunkDir, vectorDBDir] },
  },
  build: { outDir: './build', emptyOutDir: true },
  test: {
    environment: 'jsdom',
    include: ['vite/tests/**/*.test.js'],
    globals: true,
  },
})

