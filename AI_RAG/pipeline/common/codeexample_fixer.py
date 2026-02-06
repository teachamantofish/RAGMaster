# md_code_fix_dir.py
# Usage:
#   python md_code_fix_dir.py "<job_cwd>"
#
# Notes:
# - Overwrites each *.md in job_cwd.
# - Optional tools improve results:
#     pip install beautifulsoup4
#     npm i -g prettier
#     (Windows) choco install llvm   # for clang-format
#
# Handles:
#   JSON (stdlib), XML (stdlib), HTML (bs4 or Prettier),
#   JS/TS/CSS (Prettier), C/Java/C#/C++ (clang-format)
#
# Skips unknown languages but will still add a detected tag when possible.

import sys, os, re, json, shutil, subprocess
from xml.dom import minidom

# -------- fence detection --------
FENCE_RE = re.compile(r"(^```([^\n]*)\n)(.*?)(\n```)[ \t]*$", re.M | re.S)

LANG_ALIASES_OUT = {
    "js": "javascript", "jsx": "javascript",
    "ts": "typescript", "tsx": "typescript",
    "csharp": "csharp", "cs": "csharp",
    "c++": "cpp", "cxx": "cpp", "hpp": "cpp",
    "yml": "yaml",
}

PRETTIER_PARSERS = {
    "javascript": "babel",
    "typescript": "typescript",
    "css": "css",
    "html": "html",
}

CLANG_ASSUME = {
    "c": "code.c",
    "cpp": "code.cpp",
    "java": "code.java",
    "csharp": "code.cs",
}

HTML_HINTS = re.compile(r"<!doctype\s+html|<html\b|<head\b|<body\b|<div\b|<span\b", re.I)
# MIF pattern: <Tag content> with backtick strings `text`, comments # text, and > # end of Tag
MIF_HINTS = re.compile(r"<[A-Z][A-Za-z0-9]*\s[^>]*`[^`]*`[^>]*>|>\s*#\s*end\s+of\s+[A-Z][A-Za-z0-9]*|<[A-Z][A-Za-z0-9]*\s+\d+>", re.M)

# -------- utils --------
def which(cmd): return shutil.which(cmd)

def run_cmd(cmd, input_text, cwd=None):
    p = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        cwd=cwd or None,
    )
    out, err = p.communicate(input_text.encode("utf-8"), timeout=90)
    return p.returncode, out.decode("utf-8", "replace"), err.decode("utf-8", "replace")

def normalize_lang_tag(tag: str) -> str:
    tag = (tag or "").strip().lower()
    if not tag:
        return ""
    tag = tag.split()[0]  # first token only
    return LANG_ALIASES_OUT.get(tag, tag)

# -------- language detection (best-effort) --------
def detect_language(code: str) -> str:
    s = code.strip()

    # JSON - try full JSON first, then JSON fragments
    try:
        json.loads(s)
        return "json"
    except Exception:
        # Try to detect JSON fragments (partial objects/arrays)
        stripped = s.strip()
        # Check for JSON object patterns
        if (stripped.startswith('"') and '":' in stripped and 
            ('{' in stripped or '}' in stripped) and
            re.search(r'"[^"]+"\s*:\s*[{"\[\]tfn\d]', stripped)):
            return "json"
        # Check for JSON array patterns  
        if (stripped.startswith('[') and stripped.endswith(']')) or \
           (stripped.startswith('{') and not stripped.endswith('}')):
            return "json"
        pass

    # XML / HTML / MIF
    if "<" in s and ">" in s and not s.lstrip().startswith("{"):
        # Check for MIF format first
        if MIF_HINTS.search(s):
            return "mif"
        try:
            minidom.parseString(s.encode("utf-8"))
            if HTML_HINTS.search(s):
                return "html"
            return "xml"
        except Exception:
            if HTML_HINTS.search(s):
                return "html"

    # Shell script (bash/sh) - check before JS since 'export' is common in both
    if re.search(r"\b(export|echo|cd|ls|mkdir|rm|cp|mv|chmod|chown|grep|awk|sed)\b", s) or \
       re.search(r"^\s*(export\s+\w+=|#!/bin/(bash|sh))", s, re.M) or \
       re.search(r"\$\{?\w+\}?", s):  # shell variables
        return "bash"

    # JS/TS
    if re.search(r"\b(import|export|from|const|let|function|=>|console\.log)\b", s):
        if re.search(r"\b(type|interface|enum|implements|readonly)\b|:\s*[A-Z_a-z]\w*", s):
            return "typescript"
        return "javascript"

    # CSS
    if re.search(r"\b[a-zA-Z0-9\-\._#]+?\s*\{[^}]*:[^}]*\}", s) and not re.search(r"\bfunction\b", s):
        return "css"

    # C/C++/Java/C#
    if (
        re.search(r"#include\s*<", s) or
        re.search(r"\bint\s+main\s*\(", s) or
        re.search(r"\busing\s+namespace\b", s) or
        re.search(r"\bpublic\s+static\s+void\s+main\s*\(", s) or
        re.search(r"\bConsole\.WriteLine\b", s) or
        re.search(r"\bSystem\.out\.println\b", s) or
        re.search(r"\bclass\s+[A-Z_]\w*", s)
    ):
        if "#include" in s or "using namespace" in s:
            return "cpp"
        if "public static void main" in s or "System.out.println" in s:
            return "java"
        if "Console.WriteLine" in s:
            return "csharp"
        return "cpp" if re.search(r";\s*$", s, re.M) else "c"

    return ""  # unknown

# -------- formatters --------
def fmt_json(txt: str) -> str | None:
    try:
        # Try to format as complete JSON first
        return json.dumps(json.loads(txt), indent=2, ensure_ascii=False) + "\n"
    except Exception:
        # Try to format as JSON fragment by wrapping it
        stripped = txt.strip()
        if (stripped.startswith('"') and '":' in stripped and 
            ('{' in stripped or '}' in stripped)):
            try:
                # Wrap fragment in braces to make it valid JSON
                wrapped = '{' + stripped + '}'
                parsed = json.loads(wrapped)
                formatted = json.dumps(parsed, indent=2, ensure_ascii=False)
                # Remove the wrapper braces and re-indent
                lines = formatted.split('\n')
                if len(lines) > 2:
                    # Remove first { and last }
                    inner_lines = lines[1:-1]
                    # Dedent by 2 spaces
                    result_lines = []
                    for line in inner_lines:
                        if line.startswith('  '):
                            result_lines.append(line[2:])
                        else:
                            result_lines.append(line)
                    return '\n'.join(result_lines) + '\n'
            except Exception:
                pass
        return None

def fmt_xml(txt: str) -> str | None:
    try:
        dom = minidom.parseString(txt.encode("utf-8"))
        return dom.toprettyxml(indent="  ")
    except Exception:
        return None

def fmt_html(txt: str, job_cwd: str) -> str | None:
    # Use BeautifulSoup for HTML formatting (Python-only)
    try:
        from bs4 import BeautifulSoup  # type: ignore
        return BeautifulSoup(txt, "html.parser").prettify()
    except ImportError:
        print("[warn] beautifulsoup4 not installed, skipping HTML formatting")
        return None

def fmt_python_js_css(txt: str, lang: str) -> str | None:
    """Format JavaScript, CSS, and other code using Python libraries."""
    try:
        if lang in ("javascript", "typescript"):
            # Use jsbeautifier for JavaScript/TypeScript
            import jsbeautifier
            options = jsbeautifier.default_options()
            options.indent_size = 2
            return jsbeautifier.beautify(txt, options)
        elif lang == "css":
            # Use cssbeautifier for CSS
            import cssbeautifier
            return cssbeautifier.beautify(txt, {"indent_size": 2})
    except ImportError:
        print(f"[warn] jsbeautifier/cssbeautifier not installed, skipping {lang} formatting")
        return None
    except Exception as e:
        print(f"[warn] {lang} formatting failed: {e}")
        return None
    
    return None

def fmt_prettier(txt: str, lang: str, job_cwd: str) -> str | None:
    # Replaced with Python-only alternative
    return fmt_python_js_css(txt, lang)

def fmt_clang(txt: str, lang: str, job_cwd: str) -> str | None:
    exe = which("clang-format")
    if not exe:
        return None
    assume = CLANG_ASSUME.get(lang, "code.cpp")
    rc, out, _ = run_cmd([exe, "--assume-filename", assume, "-style", "file"], txt, cwd=job_cwd)
    return out if rc == 0 else None

def fmt_mif(txt: str) -> str | None:
    """Format MIF (FrameMaker Interchange Format) code with proper indentation."""
    try:
        # First, split on logical boundaries if everything is on one line
        content = txt.strip()
        
        # If the content appears to be all on one line, split it intelligently
        if '\n' not in content or content.count('\n') <= 1:
            # Split before opening tags that are not at the start
            content = re.sub(r'\s+<([A-Z][A-Za-z0-9]*)', r'\n<\1', content)
            # Split after closing tags with comments
            content = re.sub(r'>\s*(#[^<>]*?)(?=\s*[<>])', r'>\n\1\n', content)
            # Split after standalone closing tags
            content = re.sub(r'>\s*(?=<)', r'>\n', content)
            # Split complex endings with multiple comments and closures
            content = re.sub(r'>\s*#([^#]*?)#([^>]*?)\s*>', r'>\n#\1\n#\2\n>', content)
        
        lines = content.split('\n')
        formatted_lines = []
        indent_level = 0
        indent_str = "  "  # 2 spaces per level
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Handle comment-only lines differently
            if line.startswith('#'):
                formatted_lines.append(indent_str * indent_level + line)
                continue
                
            # Handle standalone closing brackets
            if line == '>':
                indent_level = max(0, indent_level - 1)
                formatted_lines.append(indent_str * indent_level + line)
                continue
                
            # Count opening and closing brackets
            opens = line.count('<')
            closes = line.count('>')
            
            # Handle closing brackets first (decrease indent before this line)
            if closes > opens:
                indent_level = max(0, indent_level - (closes - opens))
            
            # Add indented line
            formatted_lines.append(indent_str * indent_level + line)
            
            # Handle opening brackets (increase indent after this line)
            if opens > closes:
                indent_level += (opens - closes)
        
        return '\n'.join(formatted_lines)
    except Exception as e:
        print(f"[warn] MIF formatting failed: {e}")
        return None

def format_code(code: str, lang: str, job_cwd: str) -> str | None:
    if not code.endswith("\n"):
        code = code + "\n"

    try:
        if lang == "json":
            j = fmt_json(code)
            if j is not None: return j

        if lang == "xml":
            x = fmt_xml(code)
            if x is not None: return x

        if lang == "html":
            h = fmt_html(code, job_cwd)
            if h is not None: return h

        if lang == "mif":
            m = fmt_mif(code)
            if m is not None: return m

        if lang in ("javascript", "typescript", "css"):
            p = fmt_prettier(code, lang, job_cwd)
            if p is not None: return p

        if lang in ("c", "cpp", "java", "csharp"):
            c = fmt_clang(code, lang, job_cwd)
            if c is not None: return c
    except Exception as e:
        # If external tools fail, just continue without formatting
        print(f"[warn] formatting failed for {lang}: {e}")
        pass

    return None  # unknown or no formatter

# -------- markdown processing --------
def process_markdown(md_text: str, job_cwd: str) -> str:
    def repl(m):
        head, info, body, tail = m.groups()
        existing = normalize_lang_tag(info)
        lang = existing or detect_language(body)

        new_head = f"```{lang}\n" if lang else head  # add language if we detected one
        formatted = format_code(body, lang, job_cwd) if lang else None

        if formatted is None:
            # keep original content; still add language if detected & missing
            if lang and not existing:
                return new_head + body + tail
            return head + body + tail

        return new_head + formatted + tail

    out = FENCE_RE.sub(repl, md_text)
    if not out.endswith("\n"):
        out += "\n"
    return out

# -------- main --------
def process_directory(job_cwd: str) -> None:
    """
    Process all .md files in the given directory to format code blocks.
    
    Args:
        job_cwd: Directory path containing .md files to process
    """
    job_cwd = os.path.abspath(job_cwd)
    if not os.path.isdir(job_cwd):
        print(f"[err] not a directory: {job_cwd}")
        return

    md_files = sorted(f for f in os.listdir(job_cwd) if f.lower().endswith(".md"))
    if not md_files:
        print("[info] no .md files found.")
        return

    print(f"[info] processing {len(md_files)} markdown files in: {job_cwd}")
    for name in md_files:
        path = os.path.join(job_cwd, name)
        print(f"[debug] Processing: {path}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                md = f.read()
            fixed = process_markdown(md, job_cwd)
            with open(path, "w", encoding="utf-8") as f:
                f.write(fixed)
            print(f"[ok] {name}")
        except Exception as e:
            print(f"[fail] {name}: {e}")
            import traceback
            traceback.print_exc()

def main():
    if len(sys.argv) != 2:
        print("Usage: python md_code_fix_dir.py <job_cwd>")
        sys.exit(2)

    job_cwd = sys.argv[1]
    process_directory(job_cwd)

if __name__ == "__main__":
    main()
