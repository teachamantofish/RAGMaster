declare global {
  interface Window {
    __VITE_HMR_ACTIVE__?: boolean;
  }
}

window.__VITE_HMR_ACTIVE__ = true;

if (import.meta.hot) {
  import.meta.hot.accept(() => {
    // Intentionally left blank; we only need HMR wiring so Vite keeps the connection alive.
  });
}

export {};
