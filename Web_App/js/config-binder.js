/**
 * config-binder.js
 *
 * Binds a schema + parsed config model to DOM controls inside a container.
 * For each schema field, finds the matching control, sets its value, attaches
 * change listeners, and renders a help (?) icon wired to the shared modal.
 *
 * Exported API (ES module):
 *   bindConfig(container, schema, model, onHelp) → { collectValues(), validate() }
 */

/**
 * Bind schema fields to DOM controls.
 *
 * @param {HTMLElement} container - The scoped DOM container (e.g., #tab-content-chunk).
 * @param {{ fields: Array }} schema - The schema object with a `fields` array.
 * @param {Object} model - The parsed config values { key: typedValue }.
 * @param {Function} onHelp - Callback(label, helpText) to open the help modal.
 * @returns {{ collectValues: Function, validate: Function }}
 */
export function bindConfig(container, schema, model, onHelp) {
  const fieldMap = new Map();

  for (const field of schema.fields) {
    const control = findControl(container, field.key);
    if (!control) {
      console.warn(`[config-binder] No control found for key: ${field.key}`);
      continue;
    }

    fieldMap.set(field.key, { field, control });

    // Set initial value
    setControlValue(control, field, model[field.key]);

    // Read-only handling
    if (field.readOnly) {
      control.disabled = true;
      if (field.readOnlyReason) {
        control.title = field.readOnlyReason;
      }
    }

    // Change listener to keep model in sync
    if (!field.readOnly) {
      const handler = () => {
        model[field.key] = getControlValue(control, field);
      };
      control.addEventListener('sl-change', handler);
      control.addEventListener('sl-input', handler);
    }

    // Add help icon
    if (field.help) {
      addHelpIcon(control, field, onHelp);
    }
  }

  return {
    /**
     * Collect current values from all writable controls.
     * @returns {Object} { key: value }
     */
    collectValues() {
      const result = {};
      for (const [key, { field, control }] of fieldMap) {
        if (!field.readOnly) {
          result[key] = getControlValue(control, field);
        }
      }
      return result;
    },

    /**
     * Validate all writable fields against schema rules.
     * @returns {{ valid: boolean, errors: Array<{ key: string, message: string }> }}
     */
    validate() {
      const errors = [];
      for (const [key, { field, control }] of fieldMap) {
        if (field.readOnly) continue;
        const val = getControlValue(control, field);
        const err = validateField(field, val);
        if (err) {
          errors.push({ key, message: err });
          markInvalid(control, err);
        } else {
          clearInvalid(control);
        }
      }
      return { valid: errors.length === 0, errors };
    }
  };
}

/**
 * Find a control in the container by config key.
 * Tries: data-config-key attribute, then lowercased ID match.
 */
function findControl(container, key) {
  // Prefer data-config-key attribute
  let el = container.querySelector(`[data-config-key="${key}"]`);
  if (el) return el;

  // Fallback: ID matching the lowercase/underscore key
  const id = key.toLowerCase();
  el = container.querySelector(`#chunk-${id}`) || container.querySelector(`#${id}`);
  return el;
}

/**
 * Set the value of a Shoelace control based on field type.
 */
function setControlValue(control, field, value) {
  switch (field.type) {
    case 'boolean':
      control.value = value === true ? 'true' : 'false';
      break;
    case 'select':
      control.value = value != null ? String(value) : '';
      break;
    case 'number':
      control.value = value != null ? String(value) : '';
      break;
    case 'text':
    default:
      control.value = value != null ? String(value) : '';
      break;
  }
}

/**
 * Get the typed value from a Shoelace control based on field type.
 */
function getControlValue(control, field) {
  const raw = control.value;
  switch (field.type) {
    case 'boolean':
      return raw === 'true';
    case 'number': {
      if (raw === '' || raw == null) return null;
      const num = Number(raw);
      return isNaN(num) ? null : num;
    }
    case 'select':
      return raw;
    case 'text':
    default:
      return raw;
  }
}

/**
 * Validate a single field value against schema constraints.
 * @returns {string|null} Error message or null if valid.
 */
function validateField(field, value) {
  if (field.type === 'number') {
    if (value === null || value === undefined) return null; // allow None/empty
    if (typeof value !== 'number' || isNaN(value)) return `${field.label} must be a valid number.`;
    if (field.min !== undefined && value < field.min) return `${field.label} must be at least ${field.min}.`;
    if (field.max !== undefined && value > field.max) return `${field.label} must be at most ${field.max}.`;
  }
  if (field.type === 'select' && field.options) {
    const allowed = field.options.map(o => typeof o === 'string' ? o : o.value);
    if (!allowed.includes(value)) return `${field.label} must be one of: ${allowed.filter(Boolean).join(', ')}.`;
  }
  return null;
}

/**
 * Mark a control as invalid with an error message.
 */
function markInvalid(control, message) {
  control.setAttribute('data-user-invalid', '');
  control.helpText = message;
  control.classList.add('config-field-invalid');
}

/**
 * Clear invalid state from a control.
 */
function clearInvalid(control) {
  control.removeAttribute('data-user-invalid');
  control.helpText = '';
  control.classList.remove('config-field-invalid');
}

/**
 * Add a help (?) icon button next to a control, wired to the help callback.
 */
function addHelpIcon(control, field, onHelp) {
  // Avoid duplicating if already added
  const parent = control.parentElement;
  if (!parent) return;
  if (parent.querySelector('.config-help-btn')) return;

  const btn = document.createElement('sl-icon-button');
  btn.name = 'question-circle';
  btn.label = `Help for ${field.label}`;
  btn.classList.add('config-help-btn');
  btn.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (onHelp) onHelp(field.label, field.help);
  });

  // Insert after the control
  if (control.nextSibling) {
    parent.insertBefore(btn, control.nextSibling);
  } else {
    parent.appendChild(btn);
  }
}
