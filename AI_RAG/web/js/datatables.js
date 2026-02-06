(function (global) {
  'use strict';


/*
B	Buttons (e.g. Excel, PDF, Print) — from the Buttons extension
f	Filtering input (search box)
r	Processing indicator (“Processing…” message)
t	Table itself
i	Table information summary (“Showing 1 to 10 of 57 entries”)
p	Pagination control
l	Length changing dropdown (“Show 10/25/50 entries”)
*/

  const DEFAULT_CONFIG = Object.freeze({
    csvPath: null,
    httpUrl: null,
    preloadedKey: null,
    tableSelector: '#tab-content-source #dataTable',
    statusSelector: '#tab-content-source #status',
    displayName: null,
    checkBoxes: true,
    buttons: null,
    buttonsDom: '<"table-toolbar d-flex align-items-center flex-wrap gap-3"lfipB>rt',
    saveButton: false,
    saveButtonText: '<img class="save-icon" src="images/save.png" alt="Save" />',
    onSave: null,
    saveButtonAction: null,
    saveButtonClass: 'datatable-btn-function-save',
    saveButtonExtend: null,
    yaml: false,
    yamlText: '<img class="yaml-icon" src="images/yaml.png" alt="yaml" />',
    onyaml: null,
    yamlAction: null,
    yamlClass: 'datatable-btn-function-yaml',
    yamlExtend: null,
    qa: false,
    qaText: '<img class="qa-icon" src="images/qa.png" alt="Q and A" />',
    onQA: null,
    qaAction: null,
    qaClass: 'datatable-btn-function-qa',
    qaExtend: null,
    printPDF: false,
    printPDFText: '<img class="pdf-icon" src="images/pdf.png" alt="PDF" />',
    onPrintPDF: null,
    printPDFAction: null,
    printPDFClass: 'datatable-btn-print-pdf',
    printPDFExtend: 'pdfHtml5',
    exportExcel: false,
    exportExcelText: '<img class="excel-icon" src="images/excel.png" alt="Excel" />',
    onExportExcel: null,
    exportExcelAction: null,
    exportExcelClass: 'datatable-btn-export-excel',
    exportExcelExtend: 'excelHtml5',
    emailButton: false,
    emailButtonText: '<img class="email-icon" src="images/email.png" alt="email" />',
    onEmail: null,
    emailButtonAction: null,
    emailButtonClass: 'datatable-btn-email',
    emailButtonExtend: null,
    dataTableOptions: null,
    dom: '<"table-toolbar d-flex align-items-center flex-wrap gap-3"lfipB>rt',
    pageLength: 200,
    lengthMenu: [25, 50, 100, 200, 400],
    language: {
      search: 'Search:',
      searchPlaceholder: 'Case sensitive, regex OK',
      info: 'Showing _START_-_END_ of _TOTAL_',
      lengthMenu: 'Rows per page: _MENU_',
    },
  });

  const isHttpProtocol =
    typeof global.location !== 'undefined' &&
    (global.location.protocol === 'http:' || global.location.protocol === 'https:');

  const sourceTabState =
    global.sourceTabState ||
    {
      initialized: false,
      loading: false,
      waitAttempts: 0,
      config: Object.assign({}, DEFAULT_CONFIG),
    };

  if (!sourceTabState.config) {
    sourceTabState.config = Object.assign({}, DEFAULT_CONFIG);
  }

  function ensureCheckboxNamespace(table) {
    if (!table) return 'datatable';
    if (!table.dataset.checkboxNamespace) {
      const base = table.id && table.id.trim() ? table.id.trim() : '';
      const sanitized = base ? base.replace(/\s+/g, '-').toLowerCase() : `table-${Math.random().toString(36).slice(2, 10)}`;
      table.dataset.checkboxNamespace = `datatable-${sanitized}`;
    }
    return table.dataset.checkboxNamespace;
  }

  function syncHeaderCheckboxState(table) {
    if (!table) return;
    const headerCheckbox = table.querySelector('thead input.datatable-select-all');
    if (!headerCheckbox) return;

    const rowCheckboxes = Array.from(table.querySelectorAll('tbody input.datatable-row-select'));
    if (!rowCheckboxes.length) {
      headerCheckbox.checked = false;
      headerCheckbox.indeterminate = false;
      return;
    }

    const checkedCount = rowCheckboxes.filter(cb => cb.checked).length;
    headerCheckbox.checked = checkedCount === rowCheckboxes.length;
    headerCheckbox.indeterminate = checkedCount > 0 && checkedCount < rowCheckboxes.length;
  }

  function bindHeaderCheckbox(table) {
    if (!table) return;
    const headerCheckbox = table.querySelector('thead input.datatable-select-all');
    if (!headerCheckbox || headerCheckbox.dataset.bound === 'true') return;

    headerCheckbox.addEventListener('change', event => {
      const checked = !!event.target.checked;
      table.querySelectorAll('tbody input.datatable-row-select').forEach(cb => {
        cb.checked = checked;
      });
      headerCheckbox.indeterminate = false;
      syncHeaderCheckboxState(table);
    });

    headerCheckbox.dataset.bound = 'true';
  }

  function bindTableCheckboxDelegates(table) {
    if (!table || table.dataset.checkboxDelegatesBound === 'true') return;

    table.addEventListener('change', event => {
      const target = event.target;
      if (!target || !target.classList) return;
      if (target.classList.contains('datatable-row-select')) {
        syncHeaderCheckboxState(table);
      }
    });

    table.dataset.checkboxDelegatesBound = 'true';
  }

  const BUTTON_SPECS = [
    {
      key: 'saveButton',
      textKey: 'saveButtonText',
      defaultText: DEFAULT_CONFIG.saveButtonText,
      extendKey: 'saveButtonExtend',
      classKey: 'saveButtonClass',
      actionKeys: ['onSave', 'saveButtonAction'],
    },
    {
      key: 'yaml',
      textKey: 'yamlText',
      defaultText: DEFAULT_CONFIG.yamlText,
      extendKey: 'yamlExtend',
      classKey: 'yamlClass',
      actionKeys: ['onyaml', 'yamlAction'],
    },
    {
      key: 'qa',
      textKey: 'qaText',
      defaultText: DEFAULT_CONFIG.qaText,
      extendKey: 'qaExtend',
      classKey: 'qaClass',
      actionKeys: ['onQA', 'qaAction'],
    },
    {
      key: 'printPDF',
      textKey: 'printPDFText',
      defaultText: DEFAULT_CONFIG.printPDFText,
      extendKey: 'printPDFExtend',
      classKey: 'printPDFClass',
      actionKeys: ['onPrintPDF', 'printPDFAction'],
    },
    {
      key: 'exportExcel',
      textKey: 'exportExcelText',
      defaultText: DEFAULT_CONFIG.exportExcelText,
      extendKey: 'exportExcelExtend',
      classKey: 'exportExcelClass',
      actionKeys: ['onExportExcel', 'exportExcelAction'],
    },
        {
      key: 'emailButton',
      textKey: 'emailButtonText',
      defaultText: DEFAULT_CONFIG.emailButtonText,
      extendKey: 'emailButtonExtend',
      classKey: 'emailButtonClass',
      actionKeys: ['onEmail', 'emailButtonAction'],
    },
  ];

  function resolveButtonAction(config, actionKeys) {
    if (!config) return null;
    for (const key of actionKeys) {
      if (typeof config[key] === 'function') return config[key];
    }
    return null;
  }

  function buildButtons(config, context) {
    if (!config) return [];

    // Allow callers to supply a function that returns a button array.
    if (typeof config.buttons === 'function') {
      try {
        const generated = config.buttons(context);
        if (Array.isArray(generated)) {
          return generated.slice();
        }
      } catch (err) {
        console.warn('[source-tab] config.buttons function threw an error:', err);
      }
    }

    if (Array.isArray(config.buttons)) {
      return config.buttons.slice();
    }

    const buttons = [];

    const createAction = (handler, key) => {
      const handlerSuffix = key.charAt(0).toUpperCase() + key.slice(1);
      const label = key
        .replace(/([A-Z])/g, ' $1')
        .replace(/^./, c => c.toUpperCase());
      return (event, dataTable, node, buttonConfig) => {
        if (typeof handler === 'function') {
          handler({
            event,
            dataTable,
            node,
            buttonConfig,
            tableElement: context.table,
            headers: context.headers,
            rows: context.rows,
            sourceConfig: config,
          });
          return;
        }
        console.info(
          `[SourceTable] ${label} clicked. Provide a handler via config.on${handlerSuffix} or config.${key}Action.`
        );
      };
    };

    const placeholderMessages = {
      emailButton: 'Email selected clicked. Provide config.onEmail to implement.',
      functionX: 'Run script X is not implemented.',
      functionY: 'Run script Y is not implemented.',
    };

    BUTTON_SPECS.forEach(spec => {
      if (!config[spec.key]) return;

      const handler = resolveButtonAction(config, spec.actionKeys);
      const text = config[spec.textKey] || spec.defaultText;
      const extend = config[spec.extendKey];
      const className = config[spec.classKey] || '';

      const buttonConfig = {
        text,
        className,
      };

      if (extend) {
        buttonConfig.extend = extend;
      }

      if (handler) {
        buttonConfig.action = createAction(handler, spec.key);
      } else if (placeholderMessages[spec.key]) {
        buttonConfig.action = () => {
          alert(placeholderMessages[spec.key]);
        };
      }

      buttons.push(buttonConfig);
    });

    return buttons;
  }

  function derivePreloadedKey(path, fallback) {
    if (fallback) return fallback;
    if (!path) return null;
    const parts = String(path).split(/[\\/]/);
    return parts[parts.length - 1] || null;
  }

  function mergeConfig(overrides) {
    const merged = Object.assign({}, DEFAULT_CONFIG, sourceTabState.config || {}, overrides || {});
    merged.preloadedKey = derivePreloadedKey(merged.csvPath, merged.preloadedKey || DEFAULT_CONFIG.preloadedKey);
    if (!merged.displayName) {
      merged.displayName = merged.preloadedKey || merged.csvPath || 'CSV data';
    }
    if (!merged.tableSelector) merged.tableSelector = DEFAULT_CONFIG.tableSelector;
    if (!merged.statusSelector) merged.statusSelector = DEFAULT_CONFIG.statusSelector;
    return merged;
  }

  function getConfig() {
    return Object.assign({}, sourceTabState.config || DEFAULT_CONFIG);
  }

  function configure(overrides, options) {
    sourceTabState.config = mergeConfig(overrides);
    sourceTabState.waitAttempts = 0;
    if (!options || options.resetInitialized !== false) {
      sourceTabState.initialized = false;
    }
    return getConfig();
  }

  function parseCsvLine(line) {
    if (!line) return [];
    if (!/"|,/.test(line)) return [line.trim()];
    const out = [];
    let cur = '';
    let inQuotes = false;
    for (let i = 0; i < line.length; i++) {
      const ch = line[i];
      if (ch === '"') {
        if (inQuotes && line[i + 1] === '"') {
          cur += '"';
          i++;
        } else {
          inQuotes = !inQuotes;
        }
      } else if (ch === ',' && !inQuotes) {
        out.push(cur);
        cur = '';
      } else {
        cur += ch;
      }
    }
    out.push(cur);
    return out.map(s => s.trim());
  }

  function parseCsv(text) {
    const lines = (text || '')
      .split(/\r?\n/)
      .filter(l => l.trim().length > 0);
    if (lines.length === 0) return { headers: [], rows: [] };
    const headers = parseCsvLine(lines[0]);
    const rows = lines.slice(1).map(parseCsvLine);
    return { headers, rows };
  }

  function renderTable(headers, rows, cfg) {
    const config = cfg ? mergeConfig(cfg) : getConfig();
    if (!global.document) return;
    const table = config.tableSelector ? global.document.querySelector(config.tableSelector) : null;
    if (!table) return;
    const thead = table.querySelector('thead');
    const tbody = table.querySelector('tbody');
    if (!thead || !tbody) return;

    const includeCheckbox = config.checkBoxes !== false;
    const namespace = includeCheckbox ? ensureCheckboxNamespace(table) : null;

    thead.innerHTML = '';
    tbody.innerHTML = '';

    const headerRow = global.document.createElement('tr');

    if (includeCheckbox) {
      const checkboxTh = global.document.createElement('th');
      checkboxTh.className = 'datatable-checkbox-header';
      const headerCheckbox = global.document.createElement('input');
      headerCheckbox.type = 'checkbox';
      headerCheckbox.className = 'datatable-select-all';
      headerCheckbox.id = `${namespace}-select-all`;
      checkboxTh.appendChild(headerCheckbox);
      headerRow.appendChild(checkboxTh);
    }

    headers.forEach(h => {
      const th = global.document.createElement('th');
      th.textContent = h;
      headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);

    rows.forEach((row, rowIndex) => {
      const tr = global.document.createElement('tr');

      if (includeCheckbox) {
        const checkboxTd = global.document.createElement('td');
        checkboxTd.className = 'datatable-checkbox-cell';
        const rowCheckbox = global.document.createElement('input');
        rowCheckbox.type = 'checkbox';
        rowCheckbox.className = 'datatable-row-select';
        rowCheckbox.dataset.rowIndex = String(rowIndex);
        checkboxTd.appendChild(rowCheckbox);
        tr.appendChild(checkboxTd);
      }

      for (let i = 0; i < headers.length; i++) {
        const td = global.document.createElement('td');
        td.textContent = row[i] !== undefined ? row[i] : '';
        tr.appendChild(td);
      }
      tbody.appendChild(tr);
    });

    if (includeCheckbox) {
      bindTableCheckboxDelegates(table);
    }

    const jQuery = global.jQuery;
    if (jQuery && jQuery.fn && typeof jQuery.fn.DataTable === 'function') {
      try {
        const $table = jQuery(table);
        if ($table.hasClass('dataTable')) {
          $table.DataTable().destroy();
        }

        const buttonDefs = buildButtons(config, { table, headers, rows });
        const hasRequestedButtons = buttonDefs.length > 0;
        const buttonsCtor =
          hasRequestedButtons &&
          ((jQuery.fn.dataTable && jQuery.fn.dataTable.Buttons) ||
            (jQuery.fn.DataTable && jQuery.fn.DataTable.Buttons));
        const buttonsPluginAvailable = typeof buttonsCtor === 'function';

        const dataColumnIndices = includeCheckbox
          ? Array.from({ length: headers.length }, (_, index) => index + 1)
          : Array.from({ length: headers.length }, (_, index) => index);

        const columnDefs = [];
        if (includeCheckbox) {
          columnDefs.push({
            targets: 0,
            orderable: false,
            className: 'dt-center dt-compact datatable-checkbox-column',
            width: '32px',
          });
        }
        if (dataColumnIndices.length) {
          columnDefs.push({
            targets: dataColumnIndices,
            className: 'dt-left',
          });
        } else if (!includeCheckbox) {
          columnDefs.push({
            targets: '_all',
            className: 'dt-left',
          });
        }

        const dataTableOptions = {
          paging: true,
          ordering: true,
          searching: true,
          info: true,
          columnDefs,
          pageLength: config.pageLength ?? DEFAULT_CONFIG.pageLength,
          lengthMenu: config.lengthMenu ?? DEFAULT_CONFIG.lengthMenu,
          language: Object.assign({}, DEFAULT_CONFIG.language, config.language || {}),
        };

        const resolvedDom = config.dom || config.buttonsDom || DEFAULT_CONFIG.dom || DEFAULT_CONFIG.buttonsDom;

        if (hasRequestedButtons) {
          if (buttonsPluginAvailable) {
            dataTableOptions.buttons = buttonDefs;
          } else {
            console.warn(
              '[source-tab] Button configuration supplied but DataTables Buttons extension is not loaded. Skipping buttons.'
            );
          }
        }

        if (!dataTableOptions.dom) {
          dataTableOptions.dom = resolvedDom;
        }

        if (config.dataTableOptions && typeof config.dataTableOptions === 'object') {
          const { language: extraLanguage, ...rest } = config.dataTableOptions;
          if (extraLanguage && typeof extraLanguage === 'object') {
            dataTableOptions.language = Object.assign({}, dataTableOptions.language, extraLanguage);
          }
          Object.assign(dataTableOptions, rest);
        }

        const dataTableInstance = $table.DataTable(dataTableOptions);

        const applyToolbarClasses = () => {
          const wrapper = $table.closest('.dataTables_wrapper');
          if (!wrapper.length) return;
          wrapper.addClass('table-toolbar-wrapper');
          wrapper
            .find('.dt-paging, .dataTables_paginate')
            .addClass('table-toolbar-paging')
            .css('gap', '4px');
          wrapper.find('.dt-paging-button, .paginate_button').addClass('table-toolbar-page-button');
        };

        applyToolbarClasses();

        if (includeCheckbox && dataTableInstance && typeof dataTableInstance.on === 'function') {
          dataTableInstance.on('draw', () => {
            bindHeaderCheckbox(table);
            syncHeaderCheckboxState(table);
            applyToolbarClasses();
          });
        } else if (dataTableInstance && typeof dataTableInstance.on === 'function') {
          dataTableInstance.on('draw', applyToolbarClasses);
        }
      } catch (err) {
        console.warn('[source-tab] DataTable init failed:', err);
      }
    }

    if (includeCheckbox) {
      bindHeaderCheckbox(table);
      syncHeaderCheckboxState(table);
    }
  }

  const hasPywebview = () => {
    if (typeof global.hasPywebviewApi === 'function') {
      try {
        return global.hasPywebviewApi();
      } catch (_) {
        /* ignore and fall through */
      }
    }
    const candidate = global.pywebview;
    return (
      typeof candidate !== 'undefined' &&
      candidate &&
      candidate.api &&
      typeof candidate.api.load_file === 'function'
    );
  };

  async function fetchCsv(cfg) {
    const config = cfg ? mergeConfig(cfg) : getConfig();
    const preloadedKey = config.preloadedKey;

    if (global.__PRELOADED_FILES && preloadedKey && global.__PRELOADED_FILES[preloadedKey]) {
      console.log('[source-tab] using preloaded CSV for', preloadedKey);
      return global.__PRELOADED_FILES[preloadedKey];
    }

    if (hasPywebview()) {
      const csvPath = config.csvPath;
      if (csvPath) {
        try {
          const result = await global.pywebview.api.load_file(csvPath);
          if (result && result.success && typeof result.content === 'string') {
            console.log('[source-tab] fetched CSV via pywebview API:', csvPath);
            return result.content;
          }
          throw new Error(result && result.error ? result.error : 'pywebview returned no content');
        } catch (err) {
          console.warn('[source-tab] pywebview fetch failed:', err);
        }
      }
    }

    const csvPath = config.csvPath;
    if (isHttpProtocol && typeof global.fetch === 'function' && (config.httpUrl || csvPath)) {
      const candidates = (() => {
        const urls = new Set();
        if (config.httpUrl) urls.add(config.httpUrl);
        if (csvPath) {
          try {
            urls.add(new URL(csvPath, global.location.href).pathname);
          } catch (_) {
            /* ignore bad URL */
          }
          urls.add(csvPath);
        }
        return Array.from(urls).filter(Boolean);
      })();

      let lastError = null;
      for (const candidate of candidates) {
        try {
          console.log('[source-tab] fetching via HTTP', candidate);
          const response = await global.fetch(candidate, { cache: 'no-store' });
          if (!response.ok) throw new Error(`HTTP ${response.status}`);
          return await response.text();
        } catch (err) {
          lastError = err;
          console.warn('[source-tab] HTTP fetch failed for', candidate, err);
        }
      }
      if (lastError) throw lastError;
    }

    throw new Error(`Unable to load ${config.displayName || config.csvPath || 'CSV data'}`);
  }

  async function loadSourceTab(forceReload = false, overrides) {
    if (overrides) configure(overrides);
    if (!global.document) return;
    if (sourceTabState.loading) return;
    if (sourceTabState.initialized && !forceReload) return;

    const config = getConfig();
    const tableEl = config.tableSelector ? global.document.querySelector(config.tableSelector) : null;
    const statusEl = config.statusSelector ? global.document.querySelector(config.statusSelector) : null;
    if (!tableEl) {
      if (sourceTabState.waitAttempts > 20) return;
      sourceTabState.waitAttempts += 1;
      global.setTimeout(() => loadSourceTab(forceReload), 100);
      return;
    }

    const displayName = config.displayName || config.csvPath || 'CSV data';

    sourceTabState.loading = true;
    if (statusEl) statusEl.textContent = `Loading ${displayName}...`;
    try {
      const csvText = await fetchCsv(config);
      const { headers, rows } = parseCsv(csvText);
      if (headers.length === 0) {
        if (statusEl) statusEl.textContent = 'No header row found in CSV.';
        sourceTabState.initialized = true;
        return;
      }
      renderTable(headers, rows, config);
      if (statusEl) {
        statusEl.textContent = `Loaded ${rows.length} row${rows.length === 1 ? '' : 's'} from ${displayName}`;
      }
      sourceTabState.initialized = true;
      sourceTabState.waitAttempts = 0;
    } catch (err) {
      console.error('[source-tab] load failed:', err);
      if (statusEl) statusEl.textContent = `Failed to load ${displayName} - ${err.message}`;
    } finally {
      sourceTabState.loading = false;
    }
  }

  global.sourceTabState = sourceTabState;
  global.loadSourceTab = loadSourceTab;
  global.SourceTable = Object.assign(global.SourceTable || {}, {
    configure,
    getConfig,
    loadSourceTab,
    fetchCsv,
    renderTable,
    parseCsv,
    parseCsvLine,
    buildButtons,
    defaults: DEFAULT_CONFIG,
  });

  if (global && typeof global.dispatchEvent === 'function') {
    try {
      if (typeof global.CustomEvent === 'function') {
        global.dispatchEvent(
          new global.CustomEvent('source-table-ready', {
            detail: { SourceTable: global.SourceTable },
          })
        );
      } else if (typeof global.Event === 'function') {
        const evt = new global.Event('source-table-ready');
        evt.detail = { SourceTable: global.SourceTable };
        global.dispatchEvent(evt);
      }
    } catch (_) {
      /* no-op */
    }
  }
})(typeof window !== 'undefined' ? window : globalThis);
