import webview
import os

class Api:
    def load_file(self, file_path):
        """Load a file from disk and return its contents"""
        try:
            # Get the directory of the current script (pywebview folder)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            print(f"Script directory: {script_dir}")
            
            # Resolve the relative path
            full_path = os.path.join(script_dir, file_path)
            # Normalize the path
            full_path = os.path.normpath(full_path)
            print(f"Full path: {full_path}")
            
            # Check if file exists
            if not os.path.exists(full_path):
                return {'success': False, 'error': f'File not found: {full_path}'}
            
            with open(full_path, 'r', encoding='utf-8') as file:
                content = file.read()
                print(f"File loaded successfully, {len(content)} characters")
            return {'success': True, 'content': content}
        except Exception as e:
            print(f"Error loading file: {str(e)}")
            return {'success': False, 'error': str(e)}

    def save_file(self, file_path, content):
        """Persist text content to disk relative to the pywebview folder"""
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            full_path = os.path.join(script_dir, file_path)
            full_path = os.path.normpath(full_path)

            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w', encoding='utf-8') as file:
                file.write(content if content is not None else '')

            print(f"File saved successfully: {full_path}")
            return {'success': True, 'path': full_path}
        except Exception as e:
            print(f"Error saving file: {str(e)}")
            return {'success': False, 'error': str(e)}

# Get absolute path to the web directory
script_dir = os.path.dirname(os.path.abspath(__file__))
web_dir = os.path.join(script_dir, '..', 'web')
index_path = os.path.join(web_dir, 'index.html')
index_path = os.path.normpath(index_path)

print(f"Loading index from: {index_path}")

api = Api()
window = webview.create_window('AI RAG Pipeline', index_path, js_api=api, width=1400, height=900)
webview.start(debug=True)
