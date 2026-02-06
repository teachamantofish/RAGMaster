;(function (global) {
  'use strict';

  const DEFAULT_FILENAME = 'selectedconfig.csv';
  const CSV_LINE_ENDING = '\n';

  function hasPywebviewApi() {
    return (
      typeof global.pywebview !== 'undefined' &&
      global.pywebview &&
      typeof global.pywebview.api === 'object' &&
      typeof global.pywebview.api.save_file === 'function'
    );
  }

  function resolveTableElement(context) {
    if (context && context.tableElement && context.tableElement.nodeType === 1) {
      return context.tableElement;
    }
    if (
      context &&
      context.dataTable &&
      typeof context.dataTable.table === 'function'
    ) {
      const tableApi = context.dataTable.table();
      const table =
        tableApi && typeof tableApi.node === 'function' ? tableApi.node() : null;
      if (table && table.nodeType === 1) {
        return table;
      }
    }
    console.warn('[SourceActions] Unable to resolve table element for checkbox scan.');
    return null;
  }

  function getCheckedCheckboxNodes(context) {
    const tableElement = resolveTableElement(context);
    if (!tableElement) {
      return [];
    }
    return Array.from(
      tableElement.querySelectorAll('tbody input.datatable-row-select:checked')
    );
  }

  function formatCountMessage(count) {
    const noun = count === 1 ? 'checkbox' : 'checkboxes';
    return `${count} ${noun} selected`;
  }

  function getSelectedRows(context) {
    if (!context || !Array.isArray(context.rows)) {
      console.warn('[SourceActions] Context rows are required to save selections.');
      return { headers: [], rows: [] };
    }

    const checkboxes = getCheckedCheckboxNodes(context);
    if (!checkboxes.length) {
      return {
        headers: Array.isArray(context.headers) ? context.headers.slice() : [],
        rows: [],
      };
    }

    const headers = Array.isArray(context.headers) ? context.headers.slice() : [];
    const sourceRows = context.rows;
    const rows = [];

    for (const checkbox of checkboxes) {
      if (!checkbox || !checkbox.dataset || checkbox.dataset.rowIndex === undefined) {
        console.warn('[SourceActions] Checkbox missing row index; skipping.');
        continue;
      }
      const index = Number(checkbox.dataset.rowIndex);
      if (!Number.isInteger(index) || index < 0 || index >= sourceRows.length) {
        console.warn('[SourceActions] Invalid row index on checkbox; skipping.');
        continue;
      }
      const row = sourceRows[index];
      if (!Array.isArray(row)) {
        console.warn('[SourceActions] Row data is not an array; skipping.');
        continue;
      }
      rows.push(row.slice());
    }

    return { headers, rows };
  }

  function csvEscape(value) {
    if (value === null || value === undefined) return '';
    const str = String(value);
    if (!/[",\r\n]/.test(str)) return str;
    return `"${str.replace(/"/g, '""')}"`;
  }

  function buildCsvRow(cells) {
    if (!Array.isArray(cells)) return '';
    return cells.map(csvEscape).join(',');
  }

  function buildCsv(headers, rows, includeHeaders = true) {
    const lines = [];
    if (includeHeaders && Array.isArray(headers) && headers.length) {
      lines.push(buildCsvRow(headers));
    }
    if (Array.isArray(rows) && rows.length) {
      for (const row of rows) {
        lines.push(buildCsvRow(row));
      }
    }
    if (!lines.length) return '';
    return lines.join(CSV_LINE_ENDING) + CSV_LINE_ENDING;
  }

  function resolveTargetPath(context, fileName) {
    if (!context || !context.sourceConfig) {
      console.warn('[SourceActions] sourceConfig is required to resolve save target.');
      return null;
    }
    const config = context.sourceConfig;
    const basePath =
      typeof config.saveTarget === 'string'
        ? config.saveTarget
        : typeof config.csvPath === 'string'
          ? config.csvPath
          : null;
    if (!basePath) {
      console.warn('[SourceActions] Neither saveTarget nor csvPath is defined on sourceConfig.');
      return null;
    }
    const separator = basePath.includes('\\') && !basePath.includes('/') ? '\\' : '/';
    const parts = basePath.split(/[\\/]/);
    if (!parts.length) {
      console.warn('[SourceActions] Unable to derive save path components.');
      return null;
    }
    parts[parts.length - 1] = fileName;
    return parts.join(separator);
  }

  async function saveCheckRows(context, options = {}) {
    const { headers, rows } = getSelectedRows(context);
    const count = rows.length;
    const includeHeaders = options.includeHeaders !== false;
    const fileName = options.fileName || DEFAULT_FILENAME;

    if (!count) {
      return {
        saved: false,
        count: 0,
        fileName,
        path: null,
        csv: '',
        reason: 'no-selection',
      };
    }

    if (!hasPywebviewApi()) {
      console.warn('[SourceActions] pywebview save_file API is required but not available.');
      throw new Error('pywebview save_file API is not available.');
    }

    const targetPath = options.targetPath || resolveTargetPath(context, fileName);
    if (!targetPath) {
      console.warn('[SourceActions] Unable to resolve target path for selected rows.');
      throw new Error('No target path resolved for selected rows.');
    }

    const csvText = buildCsv(headers, rows, includeHeaders);
    const result = await global.pywebview.api.save_file(targetPath, csvText);
    if (!result || result.success !== true) {
      const reason = (result && result.error) || 'Unknown error';
      console.warn('[SourceActions] pywebview save_file reported failure.', reason);
      throw new Error(reason);
    }

    return {
      saved: true,
      count,
      fileName,
      path: result.path || targetPath,
      csv: csvText,
    };
  }

  function countSelectedCheckboxes(context) {
    return getCheckedCheckboxNodes(context).length;
  }

  function handleSave(context) {
    const label =
      (context &&
        context.sourceConfig &&
        (context.sourceConfig.displayName || context.sourceConfig.csvPath)) ||
      'table';

    saveCheckRows(context)
      .then(result => {
        let message;
        if (!result.saved) {
          message = `Save clicked for ${label}: no checkboxes selected.`;
        } else {
          const countMessage = formatCountMessage(result.count);
          message = `Saved ${countMessage} from ${label} to ${result.path}.`;
        }
        if (typeof global.alert === 'function') {
          global.alert(message);
        } else {
          console.info(message);
        }
      })
      .catch(err => {
        const errorMessage = `Failed to save selected rows: ${err && err.message ? err.message : err}`;
        if (typeof global.alert === 'function') {
          global.alert(errorMessage);
        }
        console.error('[SourceActions] save error', err);
      });
  }

  const exported = {
    countSelectedCheckboxes,
    handleSave,
    saveCheckRows,
  };

  global.SourceActions = Object.assign({}, global.SourceActions || {}, exported);
})(typeof window !== 'undefined' ? window : globalThis);
