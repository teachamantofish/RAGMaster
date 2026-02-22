;(function (global) {
  'use strict';

  const userAgent = typeof navigator !== 'undefined' ? (navigator.userAgent || '').toLowerCase() : '';
  const detectedPywebviewUA = userAgent.includes('pywebview');
  const detectedPywebviewGlobal = typeof global.pywebview !== 'undefined';

  const PYWEBVIEW_MAX_ATTEMPTS = 6;
  const PYWEBVIEW_RETRY_DELAY = 400;

  global.__PYWEBVIEW_RUNTIME__ =
    detectedPywebviewUA || detectedPywebviewGlobal || global.__PYWEBVIEW_RUNTIME__ === true;
  global.__PYWEBVIEW_MAX_ATTEMPTS__ =
    typeof global.__PYWEBVIEW_MAX_ATTEMPTS__ === 'number'
      ? global.__PYWEBVIEW_MAX_ATTEMPTS__
      : PYWEBVIEW_MAX_ATTEMPTS;
  global.__PYWEBVIEW_RETRY_DELAY__ =
    typeof global.__PYWEBVIEW_RETRY_DELAY__ === 'number'
      ? global.__PYWEBVIEW_RETRY_DELAY__
      : PYWEBVIEW_RETRY_DELAY;

  global.addEventListener('pywebviewready', () => {
    global.__PYWEBVIEW_RUNTIME__ = true;
  });

  function getApi() {
    const candidate = global.pywebview;
    if (!candidate || typeof candidate.api !== 'object') return null;
    return candidate.api;
  }

  function hasApiMethod(name) {
    const api = getApi();
    return !!(api && typeof api[name] === 'function');
  }

  function hasPywebviewApi(requiredMethod) {
    if (requiredMethod) return hasApiMethod(requiredMethod);
    return hasApiMethod('load_file') || hasApiMethod('save_file');
  }

  function isPywebviewRuntime() {
    return global.__PYWEBVIEW_RUNTIME__ === true;
  }

  async function invoke(name, ...args) {
    const api = getApi();
    if (!api || typeof api[name] !== 'function') {
      throw new Error(`pywebview API method not available: ${name}`);
    }
    return await api[name](...args);
  }

  async function loadFile(path) {
    return await invoke('load_file', path);
  }

  async function saveFile(path, content) {
    return await invoke('save_file', path, content);
  }

  const AppBridge = {
    isPywebviewRuntime,
    hasPywebviewApi,
    invoke,
    loadFile,
    saveFile,
  };

  global.AppBridge = Object.assign({}, global.AppBridge || {}, AppBridge);
  global.hasPywebviewApi = hasPywebviewApi;
})(typeof window !== 'undefined' ? window : globalThis);
