/**
 * help-modal.js
 *
 * Shared help modal for displaying field-level descriptions.
 * Uses a single <sl-dialog id="help-dialog"> in the page shell.
 *
 * Exported API (ES module):
 *   showHelp(title, text)  — open the dialog with the given content
 *   initHelpModal()        — ensure the dialog element exists in the DOM
 */

const DIALOG_ID = 'help-dialog';

/**
 * Ensure the shared <sl-dialog> exists in the DOM.
 * Call once at app startup or before first use.
 */
export function initHelpModal() {
  if (document.getElementById(DIALOG_ID)) return;

  const dialog = document.createElement('sl-dialog');
  dialog.id = DIALOG_ID;
  dialog.label = 'Help';
  dialog.classList.add('help-dialog');

  const body = document.createElement('div');
  body.id = 'help-dialog-body';
  body.classList.add('help-dialog-body');
  dialog.appendChild(body);

  document.body.appendChild(dialog);
}

/**
 * Open the help dialog with the given title and text.
 * @param {string} title - Dialog title / field label.
 * @param {string} text - Help description text.
 */
export function showHelp(title, text) {
  initHelpModal();
  const dialog = document.getElementById(DIALOG_ID);
  if (!dialog) return;

  dialog.label = title || 'Help';
  const body = dialog.querySelector('#help-dialog-body');
  if (body) {
    // Convert newlines to <br> for basic formatting
    body.innerHTML = escapeHtml(text || '').replace(/\n/g, '<br>');
  }
  dialog.show();
}

/**
 * Basic HTML escaping for safety.
 */
function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}
