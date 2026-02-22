"""
This is an example code object to friendly-name mapper for API and SDK documentation. 
Items like KBD_KERNLEFT and PgfMarkedForNamedDestination need to be expanded into plain
English and added to the chunk file prior to embedding. This improves both simple and 
dense search prior to training models.

codename_to_friendlyname.py  --  FDK Code Name -> Friendly Name Mapper

Scans a_chunks.json for FDK constants (FV_, FP_, FA_, ...) and CamelCase API
identifiers (FirstPgfInFlow, MakeTblSelection, ...).  Auto-generates English
friendly names using an abbreviation dictionary and writes a CSV for review.

After review, run with --apply to add an 'apifriendlynames' field to each chunk.

Usage:
    python codename_to_friendlyname.py              # Extract codenames -> CSV
    python codename_to_friendlyname.py --apply       # Apply approved CSV -> chunks JSON
"""

import argparse
import csv
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from matplotlib.pylab import full, stack

###################################################
# TODO: Add logging and standard CWD lookup. 
###################################################

# ── Paths ─────────────────────────────────────────────────────────────────────
CHUNKS_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "Data" / "framemaker" / "mif_jsx" / "a_chunks.json"
)
CSV_PATH = Path(__file__).resolve().parent / "codename_friendlyname.csv"

# ── Abbreviation dictionary (FDK-specific) ────────────────────────────────────
# Keys are title-cased word segments.  Values are lowercase expansions.
ABBREVIATIONS = {
    # Object / entity types
    "Tbl":   "table",
    "Pgf":   "paragraph",
    "Doc":   "document",
    "Fmt":   "format",
    "Fl":    "flow",
    "Obj":   "object",
    "Elem":  "element",
    "Attr":  "attribute",
    "Attrval": "attribute value",
    "Cond":  "condition",
    "Cb":    "change bar",
    "Cn":    "condition",
    "Xref":  "cross reference (XREF)",
    "Ref":   "reference",
    "Var":   "variable",
    "Mkr":   "marker",
    "Char":  "character",
    "Pg":    "page",
    "Fn":    "footnote",
    "Gfx":   "graphic",
    "Dlg":   "dialog",
    "Btn":   "button",
    "Cmd":   "command",
    "Hdr":   "header",
    "Ftr":   "footer",
    "Bk":    "book",
    "Rect":  "rectangle",
    "Stk":   "strikethrough",
    "Rubi":  "rubi",
    "Comb":  "combined",
    "Tsume": "tsume",
    "dict": "dictionary",
    "cat": "catelog",

    # Property / value abbreviations
    "Prop":  "property",
    "Props": "properties",
    "Val":   "value",
    "Vals":  "values",
    "Num":   "number",
    "Idx":   "index",
    "Cnt":   "count",
    "Len":   "length",
    "Pos":   "position",
    "Loc":   "location",
    "Rng":   "range",
    "Sel":   "selection",
    "Prev":  "previous",
    "Nxt":   "next",
    "Beg":   "beginning",
    "Ldr":   "leader",
    "Dpi":   "dots per inch",
    "Pct":   "percent",
    "Wd":    "width",
    "Ht":    "height",
    "Wt":    "weight",
    "Sz":    "size",
    "Bg":    "background",
    "Fg":    "foreground",
    "Src":   "source",
    "Dst":   "destination",
    "Cols":  "columns",
    "Col":   "column",
    "Ltr":   "left to right",
    "Rtl":   "right to left",
    "Chg":   "change",
    "Param": "parameter",
    "Params":"parameters",
    "Cfg":   "configuration",
    "Ctx":   "context",
    "Clr":   "color",
    "Sep":   "separator",
    "Str":   "string",
    "Int":   "integer",
    "Uint":  "unsigned integer",
    "Nbr":   "number",
    "Dfs":   "depth first search",

    # Additional entity / concept abbreviations
    "Ti":    "text inset",
    "Cms":   "content management system",
    "Dup":   "duplicate",
    "Def":   "definition",
    "Inv":   "invalid",
    "Spec":  "specification",
    "Expr":  "expression",
    "Xrefs": "cross references",
    "Res":   "resolution",
    "Sp":    "space",
    "Eqn":   "equation",
    "Csr":   "cursor",
    "Topicsetrefs": "topic set references",
    "Structapp":    "structured application",
    "Notrim":       "no trim",
    "Charbwd":      "character backward",
    "Charfwd":      "character forward",
    "Eol":   "end of line",
    "Eow":   "end of word",
    "Eos":   "end of sentence",
    "Ele":   "element",
    "Coltop":"column top",
    "Updateall":    "update all",

    # Action abbreviations
    "Init":  "initialize",
    "Del":   "delete",
    "Ins":   "insert",
    "Upd":   "update",
    "Sync":  "synchronized",
    "Notif": "notification",
}

# CamelCase words to skip (product names, JS built-ins, etc.)
CAMELCASE_EXCLUSIONS = {
    # Product / language names
    "JavaScript", "TypeScript", "ExtendScript", "FrameMaker",
    "InDesign", "InCopy", "PostScript", "TrueType", "OpenType",
    "PageMaker", "PowerPoint", "WordPerfect", "QuarkXPress",
    "CamelCase", "PascalCase", "MacOS", "AppKit",
    # JS built-ins
    "TypeError", "RangeError", "SyntaxError", "ReferenceError",
    "ArrayBuffer", "DataView", "RegExp", "StringT", "ObjHandleT",
    "forEach", "indexOf", "lastIndexOf",
    "toString", "valueOf", "parseInt", "parseFloat", "isNaN",
    "hasOwnProperty", "isPrototypeOf",
}

# ── Regex patterns ────────────────────────────────────────────────────────────
# Prefixed FDK constants (with optional Constants. accessor)
RE_PREFIX = re.compile(
    r"(?:Constants\.)?(?:FV|FP|FA|FO|FS|FF|FE|FTI|FT)_[A-Za-z0-9_]+"
)
# F_Api functions
RE_FAPI = re.compile(r"\bF_Api[A-Za-z]+\b")
# F_...T structs (e.g. F_AttributeT, F_TextLocT)
RE_STRUCT = re.compile(r"\bF_[A-Z][a-zA-Z]+T\b")
# FCodes keyboard codes
RE_FCODES = re.compile(r"FCodes\.[A-Za-z0-9_]+")
# CamelCase / PascalCase identifiers (2+ word segments, min 5 chars)
RE_CAMEL = re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*){1,}\b")


# ══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════════════════════

def load_chunks() -> list[dict]:
    """Load chunks from JSON file."""
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_codename(raw: str) -> str:
    """Strip accessor prefixes to get canonical codename.

    Constants.FV_COLOR_CYAN  ->  FV_COLOR_CYAN
    FCodes.KBD_FOOTNOTE      ->  KBD_FOOTNOTE
    """
    if raw.startswith("Constants."):
        return raw[len("Constants."):]
    if raw.startswith("FCodes."):
        return raw[len("FCodes."):]
    return raw


def split_camelcase(name: str) -> list[str]:
    """Split CamelCase/PascalCase into word segments.

    FirstPgfInFlow  ->  ['First', 'Pgf', 'In', 'Flow']
    TblNumCols      ->  ['Tbl', 'Num', 'Cols']
    HTMLParser      ->  ['HTML', 'Parser']
    """
    return re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z0-9]+|[A-Z]+", name)


def expand_codename(codename: str) -> str:
    """Convert FDK code name to a friendly English phrase.

    FV_TBL_TITLE_ABOVE  ->  table title above
    FP_FirstPgfInFlow   ->  first paragraph in flow
    F_ApiGetId          ->  get id
    F_AttributeT        ->  attribute
    KBD_FOOTNOTE        ->  footnote
    """
    name = codename

    # 1. Strip accessor prefixes (already normalized, but just in case)
    name = re.sub(r"^Constants\.", "", name)
    name = re.sub(r"^FCodes\.", "", name)

    # 2. Handle specific patterns
    if name.startswith("F_Api"):
        name = name[5:]                           # strip 'F_Api'
    elif re.match(r"^F_[A-Z].*T$", name) and not name.startswith("FV_"):
        name = name[2:-1]                         # strip 'F_' prefix and trailing 'T'
    else:
        name = re.sub(
            r"^(?:FV|FP|FA|FO|FS|FF|FE|FTI|FT|KBD)_", "", name
        )

    if not name:
        return codename.lower()

    # 3. Split on underscores, then CamelCase within each segment
    segments = name.split("_")
    words = []

    for seg in segments:
        if not seg:
            continue
        if seg.isupper() and len(seg) > 1:
            # ALL_CAPS segment: title-case for abbreviation lookup
            titled = seg.title()
            words.append(ABBREVIATIONS.get(titled, seg.lower()))
        else:
            # CamelCase or mixed
            parts = split_camelcase(seg)
            for part in parts:
                if part in ABBREVIATIONS:
                    words.append(ABBREVIATIONS[part])
                elif part.title() in ABBREVIATIONS:
                    words.append(ABBREVIATIONS[part.title()])
                else:
                    words.append(part.lower())

    return " ".join(words)


def classify_codename(codename: str) -> str:
    """Classify a codename into a type category for CSV grouping."""
    prefixes = {
        "FV_": "value",   "FP_": "property", "FA_": "action",
        "FO_": "object",  "FS_": "setting",  "FF_": "flag",
        "FE_": "error",   "FTI_": "text_item","FT_": "type",
    }
    for pfx, label in prefixes.items():
        if codename.startswith(pfx):
            return label
    if codename.startswith("F_Api"):
        return "function"
    if re.match(r"^F_[A-Z].*T$", codename):
        return "struct"
    if codename.startswith("KBD_"):
        return "keyboard"
    return "identifier"


# ══════════════════════════════════════════════════════════════════════════════
# EXTRACT mode  --  scan chunks, write CSV
# ══════════════════════════════════════════════════════════════════════════════

def extract_codenames(chunks: list[dict]) -> dict[str, dict]:
    """Extract all FDK constants and CamelCase identifiers from chunk content.

    Returns: {canonical_codename: {'count': int, 'type': str}}
    """
    results: dict[str, dict] = {}

    for chunk in chunks:
        content = chunk.get("content", "")
        found_in_chunk: set[str] = set()

        # ── Prefix-based patterns (track spans to avoid CamelCase overlap) ──
        prefix_spans: set[tuple[int, int]] = set()

        for pattern in (RE_PREFIX, RE_FAPI, RE_STRUCT, RE_FCODES):
            for m in pattern.finditer(content):
                prefix_spans.add((m.start(), m.end()))
                canonical = normalize_codename(m.group())
                found_in_chunk.add(canonical)

        # ── CamelCase identifiers (only if NOT inside a prefix match) ──
        for m in RE_CAMEL.finditer(content):
            # Skip if this span is contained within any prefix match
            if any(ps <= m.start() and m.end() <= pe for ps, pe in prefix_spans):
                continue
            name = m.group()
            if name in CAMELCASE_EXCLUSIONS or len(name) < 5:
                continue
            # Skip if this is a prefix substring of an excluded word
            # (catches crawl artifacts like "FrameMa" from "FrameMa ker")
            if any(exc.startswith(name) and exc != name for exc in CAMELCASE_EXCLUSIONS):
                continue
            found_in_chunk.add(name)

        # ── Accumulate counts ──
        for codename in found_in_chunk:
            if codename not in results:
                results[codename] = {
                    "count": 0,
                    "type": classify_codename(codename),
                }
            results[codename]["count"] += 1

    return results


def write_csv(codenames: dict[str, dict]) -> None:
    """Write extraction results to CSV for human review."""
    rows = []
    for codename, info in codenames.items():
        rows.append({
            "count":        info["count"],
            "type":         info["type"],
            "codename":     codename,
            "friendlyname": expand_codename(codename),
            "review":       "",
        })

    # Sort: type, then count descending, then alphabetical
    rows.sort(key=lambda r: (r["type"], -r["count"], r["codename"]))

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["count", "type", "codename", "friendlyname", "review"]
        )
        writer.writeheader()
        writer.writerows(rows)

    # ── Summary ──
    type_counts = Counter(r["type"] for r in rows)
    print(f"\nWrote {len(rows)} codenames to {CSV_PATH}")
    print(f"\nBreakdown by type:")
    for t, c in sorted(type_counts.items()):
        print(f"  {t:15s} {c:5d}")
    print(f"  {'TOTAL':15s} {len(rows):5d}")


# ══════════════════════════════════════════════════════════════════════════════
# APPLY mode  --  read approved CSV, enrich chunks
# ══════════════════════════════════════════════════════════════════════════════

def apply_csv(chunks: list[dict]) -> list[dict]:
    """Read approved CSV and add 'apifriendlynames' field to each chunk."""

    # 1. Load approved mappings  (skip rows with blank friendlyname)
    mappings: dict[str, str] = {}
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cn = row["codename"].strip()
            fn = row["friendlyname"].strip()
            if cn and fn:
                mappings[cn] = fn

    print(f"Loaded {len(mappings)} approved mappings from {CSV_PATH}")

    # 2. Build search forms  (canonical + Constants./FCodes. variants)
    search_to_canonical: dict[str, str] = {}
    for cn in mappings:
        search_to_canonical[cn] = cn
        if cn.startswith(("FV_", "FP_", "FA_", "FO_", "FS_",
                          "FF_", "FE_", "FTI_", "FT_")):
            search_to_canonical[f"Constants.{cn}"] = cn
        if cn.startswith("KBD_"):
            search_to_canonical[f"FCodes.{cn}"] = cn

    # 3. Compile regex  (longest first to prevent partial matches)
    sorted_forms = sorted(search_to_canonical, key=len, reverse=True)
    pattern = re.compile("|".join(re.escape(f) for f in sorted_forms))

    # 4. Apply to each chunk
    enriched_count = 0
    for chunk in chunks:
        content = chunk.get("content", "")
        found_canonical: set[str] = set()
        for m in pattern.finditer(content):
            found_canonical.add(search_to_canonical[m.group()])

        if found_canonical:
            pairs = sorted(
                f"{cn}: {mappings[cn]}" for cn in found_canonical
            )
            chunk["apifriendlynames"] = "; ".join(pairs)
            enriched_count += 1
        else:
            chunk["apifriendlynames"] = ""

    print(f"Enriched {enriched_count}/{len(chunks)} chunks with apifriendlynames")
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# CONVERT mode  --  expand friendly names to plain English, write output file
# ══════════════════════════════════════════════════════════════════════════════

# Full-phrase overrides: if the entire friendlyname matches a key here,
# use the value verbatim instead of running the word-level expander.


PLAIN_ENGLISH_OVERRIDES = {
    "newformat": "new format",
    "textsel": "text selection",
    "props": "properties",
    "designkit": "design kit",
    "dblunderline": "double underline",
    "lessstretch": "less stretch",
    "txt": "text",
    "morestretch": "more stretch",
    "normalcase": "normal case",
    "allcaps": "all caps",
    "nostretch": "no stretch",
    "halfwidth": "half width",
    "numunderline": "number underline",
    "fullwidth": "full width",
    "seldecsize": "selection decrease size",
    "selincsize": "selection increase size",
    "inc": "increase",
    "selitalic": "selection italic",
    "selroman": "selection roman",
    "selplain": "selection plain",
    "decrease": "decrease",
    "selunderline": "selection underline",
    "sel": "selection",
    "dir": "direction",
    "vert": "vertical",
    "pagehasaLeftorRight": "page has a left or right",
    "ThesourcefileisaMIF": "the source file is a MIF",
    "ti": "text inset",
    "para": "paragraph",
    "horiz": "horizontal",
    "win": "window",
    "corporatenews": "corporate news",
    "aboutproduct": "about product",
    "acceptallchanges": "accept all changes",
    "acceptchange": "accept change",
    "acceptchangeandnext": "accept change and next",
    "acceptchangeandprev": "accept change and previous",
    "addautocorrect": "add autocorrect",
    "autocorrect": "auto correct",
    "adddocdict": "add document dictionary",
    "addmarkertype": "add marker type",
    "addpage": "add page",
    "addusrdict": "add user dictionary",
    "arctool": "arc tool",
    "asc adddocdict": "asc add document dictionary",
    "asc addusrdict": "asc add user dictionary",
    "asc enable auto spell": "asc enable auto spell",
    "atomize inset": "atomize inset",
    "attrcond": "attribute condition",
    "attreditquick": "attribute edit quick",
    "attribute edit": "attribute edit",
    "attribute config file maker": "attribute config file maker",
    "attribute disp opts": "attribute display options",
    "back": "back",
    "backstack": "back stack",
    "backtab": "back tab",
    "bodypage": "body page",
    "bookaddfile": "book add file",
    "bookaddfolder": "book add folder",
    "bookaddgroup": "book add group",
    "bookcomp exclude": "book component exclude",
    "bookcomp filename": "book component file name",
    "bookcomp text": "book component text",
    "bookdisplayfilename": "book display file name",
    "bookdisplaytext": "book display text",
    "bookeditdefine": "book edit define",
    "bookrenamefile": "book rename file",
    "cbarpro": "change bar properties",
    "cellfmtquick": "cell format quick",
    "changedict": "change dictionary",
    "changequick": "change quick",
    "charfmt delete": "character format delete",
    "charfmt delete unuse": "character format delete unused",
    "checkbatch": "check batch",
    "checkdoc": "check document",
    "checkpage": "check page",
    "checksel": "check selection",
    "clear": "clear",
    "clearauto": "clear auto",
    "clopwindow": "close open window",
    "closewindow": "close window",
    "close attreditor": "close attribute editor",
    "close character catal": "close character catalog",
    "close character desig": "close character designer",
    "close custrs": "close custom ruler set",
    "close equations palett": "close equations palette",
    "close fontfmt": "close font format",
    "close paragraph catal": "close paragraph catalog",
    "close paragraph desig": "close paragraph designer",
    "close pgffmt": "close paragraph format",
    "close strwindow": "close structure window",
    "close tblfmt": "close table format",
    "cms connection mgr": "CMS connection manager",
    "condindicator": "condition indicator",
    "condinquick": "condition in quick",
    "condnotinquick": "condition not in quick",
    "condtext": "condition text",
    "condtoggleoverr": "condition toggle override",
    "condvisibility": "condition visibility",
    "condvisonlyquick": "condition visible only quick",
    "contextmenu": "context menu",
    "copyattrs": "copy attributes",
    "copycellfmt": "copy cell format",
    "copycolw": "copy column width",
    "copycond": "copy condition",
    "copyfont": "copy font",
    "copypgf": "copy paragraph",
    "cust text frame": "custom text frame",
    "cutboth": "cut both",
    "cuthead": "cut head",
    "dashoption": "dash option",
    "decdash": "decrease dash",
    "decfill": "decrease fill",
    "decpen": "decrease pen",
    "decwidth": "decrease width",
    "deldocdict": "delete document dictionary",
    "deletemarker": "delete marker",
    "deletepage": "delete page",
    "delinkhotspot": "delink hotspot",
    "delmarkertype": "delete marker type",
    "delusrdict": "delete user dictionary",
    "derefref": "dereference reference",
    "docinfo": "document info",
    "dsexit": "document set exit",
    "dumphypertext": "dump hypertext",
    "ecapture": "e-capture",
    "edithotspot": "edit hotspot",
    "editlinks": "edit links",
    "editmarkertype": "edit marker type",
    "editvariable": "edit variable",
    "elementwindow": "element window",
    "emspace": "em space",
    "enspace": "en space",
    "exposewindow": "expose window",
    "findnext": "find next",
    "findprev": "find previous",
    "firstpage": "first page",
    "firsttab": "first tab",
    "fliplr": "flip left right",
    "flipud": "flip up down",
    "fmpip": "FrameMaker pip",
    "fnext": "find next",
    "fontdesign": "font designer",
    "fontpod": "font pod",
    "fontquick": "font quick",
    "fontreplacepod": "font replace pod",
    "fontwindow": "font window",
    "footnote": "footnote",
    "footnotepro": "footnote properties",
    "fprev": "find previous",
    "frametool": "frame tool",
    "freetool": "free tool",
    "fullrulers": "full rulers",
    "gbl end": "global end",
    "gbl start": "global start",
    "generate": "generate",
    "gettrigger": "get trigger",
    "gotoip": "go to insertion point",
    "gotolinen": "go to line number",
    "gotopage": "go to page",
    "gotopagen": "go to page number",
    "hardhyphen": "hard hyphen",
    "hardreturn": "hard return",
    "hardspace": "hard space",
    "heatref": "heat reference",
    "help onlinemanuals": "help online manuals",
    "hide hotspotindicators": "hide hotspot indicators",
    "hishwindow": "history window",
    "hist": "history",
    "hotspot": "hotspot",
    "hotspotindicators": "hotspot indicators",
    "hotspotspod": "hotspots pod",
    "hscroll": "horizontal scroll",
    "hyprtxt shtcut": "hypertext shortcut",
    "incdash": "increment dash",
    "incfill": "increment fill",
    "incpen": "increment pen",
    "incwidth": "increment width",
    "indecross reference (XREF)s": "index cross references",
    "initcap": "initial capitals",
    "initcaph": "initial capitals here",
    "inline attredtr": "inline attribute editor",
    "input": "input",
    "insertobject": "insert object",
    "insertquick": "insert quick",
    "insetpod": "inset pod",
    "joincurves": "join curves",
    "kbmacro": "keyboard macro",
    "keeptool": "keep tool",
    "kerndown": "kern down",
    "kerndown6": "kern down 6",
    "kernhome": "kern home",
    "kernleft": "kern left",
    "kernleft6": "kern left 6",
    "kernright": "kern right",
    "kernright6": "kern right 6",
    "kernup": "kern up",
    "kernup6": "kern up 6",
    "lastpage": "last page",
    "lasttool": "last tool",
    "lgeqn": "large equation",
    "linelayout": "line layout",
    "linenumpro": "line number properties",
    "linenumtoggle": "line number toggle",
    "linetool": "line tool",
    "mancond": "manage conditions",
    "markerspod": "markers pod",
    "masterpage": "master page",
    "mathwindow": "math window",
    "medeqn": "medium equation",
    "memfail": "memory failure",
    "mem stats": "memory statistics",
    "menubarfocus": "menu bar focus",
    "menucomplete": "menu complete",
    "menucustom": "menu custom",
    "menumodify": "menu modify",
    "menuquick": "menu quick",
    "menureset": "menu reset",
    "minimize": "minimize",
    "mode rotate tool": "mode rotate tool",
    "movewindow": "move window",
    "newaframe": "new anchored frame",
    "newhypertext": "new hypertext",
    "newmarker": "new marker",
    "newmaster": "new master page",
    "newvar": "new variable",
    "nextpage": "next page",
    "nochangedb": "no change bar",
    "normalize tags": "normalize tags",
    "numlock": "num lock",
    "numspace": "numeric space",
    "obalign bottom": "object align bottom",
    "obalign center": "object align center",
    "obalign left": "object align left",
    "obalign middle": "object align middle",
    "obalign right": "object align right",
    "obalign top": "object align top",
    "objdown": "object down",
    "objleft": "object left",
    "objprops": "object properties",
    "objright": "object right",
    "objselect": "object select",
    "objselect nopref": "object select no preference",
    "objup": "object up",
    "openall": "open all",
    "openline": "open line",
    "openwindow": "open window",
    "open in popup win in": "open in popup window",
    "ovaltool": "oval tool",
    "pageback": "page back",
    "pagebreak": "page break",
    "pagelayout": "page layout",
    "pagesetup": "page setup",
    "pagesize": "page size",
    "pagestatus": "page status",
    "pageupdate": "page update",
    "pastespecial": "paste special",
    "pgfdesign": "paragraph design",
    "pgffmt delete": "paragraph format delete",
    "pgffmt delete unused": "paragraph format delete unused",
    "pgfquick": "paragraph quick",
    "pgfwindow": "paragraph window",
    "pickobjprops": "pick object properties",
    "podlocation": "pod location",
    "polygtool": "polygon tool",
    "polyltool": "polyline tool",
    "previewfba": "preview frame-based authoring",
    "preview acceptall": "preview accept all",
    "preview off": "preview off",
    "preview rejectall": "preview reject all",
    "prevpage": "previous page",
    "printsetup": "print setup",
    "putinline": "put inline",
    "quietclosewindow": "quiet close window",
    "quitall": "quit all",
    "quitwindow": "quit window",
    "randf": "replace and find",
    "recttool": "rectangle tool",
    "reference": "reference",
    "reformatdoc": "reformat document",
    "refpage": "reference page",
    "refreshwindow": "refresh window",
    "refresh ditamap rmvie": "refresh DITA map resource manager view",
    "rejectallchange": "reject all changes",
    "rejectchange": "reject change",
    "rejectchangeandnext": "reject change and next",
    "removeposter": "remove poster",
    "remove struct": "remove structure",
    "renameframe": "rename frame",
    "renamemarkertype": "rename marker type",
    "renameorplain": "rename or plain",
    "renamepage": "rename page",
    "reordermaster": "reorder master page",
    "repeatnew": "repeat new",
    "rerotate": "re-rotate",
    "resetdb": "reset database",
    "reshape": "reshape",
    "resizebox": "resize box",
    "resizeboxm": "resize box margins",
    "resizelock": "resize lock",
    "resizeunlock": "resize unlock",
    "restorefont": "restore font",
    "rglobal": "replace global",
    "rm mode": "review mode",
    "ronce": "replace once",
    "rotate ccw": "rotate counterclockwise",
    "rotate ccw small": "rotate counterclockwise small",
    "rotate cw": "rotate clockwise",
    "rotate cw small": "rotate clockwise small",
    "rotpage minus": "rotate page minus",
    "rotpage norm": "rotate page normal",
    "rotpage plus": "rotate page plus",
    "rot minus": "rotate minus",
    "rot plus": "rotate plus",
    "roundrect": "round rectangle",
    "rubiprops": "rubi properties",
    "saveall": "save all",
    "saveas": "save as",
    "saveasdbre": "save as database",
    "saveaspdf": "save as PDF",
    "saveaspdfreview": "save as PDF review",
    "saveaspdfreview2": "save as PDF review 2",
    "saveaspdfshare": "save as PDF share",
    "saveaspdfubiq": "save as PDF ubiquitous",
    "saveasxml": "save as XML",
    "savedbre": "save database",
    "savefmx": "save FrameMaker XML",
    "savemeta": "save metadata",
    "savesas": "save as",
    "searchreferences": "search references",
    "selectall": "select all",
    "select chapter compo": "select chapter composition",
    "setalign properties": "set align properties",
    "setcap": "set line cap style",
    "setcap 0": "set line cap style 0",
    "setcap 1": "set line cap style 1",
    "setcap 2": "set line cap style 2",
    "setcap 3": "set line cap style 3",
    "setcap option": "set line cap option",
    "setdash": "set dash pattern",
    "setdash 0": "set dash pattern 0",
    "setdash 1": "set dash pattern 1",
    "setdash 2": "set dash pattern 2",
    "setdash 3": "set dash pattern 3",
    "setdash 4": "set dash pattern 4",
    "setdash 5": "set dash pattern 5",
    "setdash 6": "set dash pattern 6",
    "setdash 7": "set dash pattern 7",
    "setdash 8": "set dash pattern 8",
    "setdash option": "set dash pattern option",
    "setdistribute properties": "set distribute properties",
    "setelcatall": "set element catalog all",
    "setelcatchild": "set element catalog child",
    "setelcatfreq": "set element catalog frequent",
    "setelcatloose": "set element catalog loose",
    "setelcatstrict": "set element catalog strict",
    "setfill": "set fill pattern",
    "setfill 0": "set fill pattern 0",
    "setfill 1": "set fill pattern 1",
    "setfill 2": "set fill pattern 2",
    "setfill 3": "set fill pattern 3",
    "setfill 4": "set fill pattern 4",
    "setfill 5": "set fill pattern 5",
    "setfill 6": "set fill pattern 6",
    "setfill 7": "set fill pattern 7",
    "setfill 8": "set fill pattern 8",
    "setfill 9": "set fill pattern 9",
    "setfill a": "set fill pattern A",
    "setfill b": "set fill pattern B",
    "setfill c": "set fill pattern C",
    "setfill d": "set fill pattern D",
    "setfill e": "set fill pattern E",
    "setfill f": "set fill pattern F",
    "setfromcolor": "set from color",
    "setknockout": "set knockout",
    "setoverprint": "set overprint",
    "setpen": "set pen pattern",
    "setpen 0": "set pen pattern 0",
    "setpen 1": "set pen pattern 1",
    "setpen 2": "set pen pattern 2",
    "setpen 3": "set pen pattern 3",
    "setpen 4": "set pen pattern 4",
    "setpen 5": "set pen pattern 5",
    "setpen 6": "set pen pattern 6",
    "setpen 7": "set pen pattern 7",
    "setpen 8": "set pen pattern 8",
    "setpen 9": "set pen pattern 9",
    "setpen a": "set pen pattern A",
    "setpen b": "set pen pattern B",
    "setpen c": "set pen pattern C",
    "setpen d": "set pen pattern D",
    "setpen e": "set pen pattern E",
    "setpen f": "set pen pattern F",
    "setposter": "set poster",
    "setrun properties": "set run-in properties",
    "setsearch": "set search",
    "setsep": "set color separation",
    "setsep all": "set color separation all",
    "setsep keep": "set color separation keep",
    "setsep reset tint over": "set color separation reset tint overprint",
    "setsides": "set sides",
    "setsolid": "set solid",
    "settint": "set tint",
    "setwidth": "set width",
    "setwidth 0": "set width 0",
    "setwidth 1": "set width 1",
    "setwidth 2": "set width 2",
    "setwidth 3": "set width 3",
    "setwidth option": "set width option",
    "setwidth slide": "set width slide",
    "set textframe grid": "set text frame grid",
    "shftspace": "shift space",
    "shownext": "show next",
    "showprev": "show previous",
    "show borders": "show borders",
    "show condition ind": "show condition indicator",
    "show element context av": "show element context availability",
    "show hotspotindicator": "show hotspot indicator",
    "smalltoolwindow": "small tool window",
    "smeqn": "small equation",
    "softhyphen": "soft hyphen",
    "spellreset": "spelling reset",
    "splitl": "split left",
    "splitr": "split right",
    "spoptions": "spelling options",
    "strip flowstructure": "strip flow structure",
    "strwindow": "structure window",
    "strwin leftanchor": "structure window left anchor",
    "stuff item": "stuff item",
    "stylefmt delete": "style format delete",
    "stylefmt delete unuse": "style format delete unused",
    "symfont": "symbol font",
    "tablewindow": "table window",
    "table addrc": "table add rows and columns",
    "table cellfmt": "table cell format",
    "table exit ip": "table exit insertion point",
    "table resizecol": "table resize column",
    "table rowfmt": "table row format",
    "tagstatus": "tag status",
    "tblfmt delete": "table format delete",
    "tblfmt delete unused": "table format delete unused",
    "tblip above": "table insertion point above",
    "tblip below": "table insertion point below",
    "tblip bottom": "table insertion point bottom",
    "tblip left": "table insertion point left",
    "tblip leftmost": "table insertion point leftmost",
    "tblip next": "table insertion point next",
    "tblip previous": "table insertion point previous",
    "tblip right": "table insertion point right",
    "tblip rightmost": "table insertion point rightmost",
    "tblip top": "table insertion point top",
    "tblip topleft": "table insertion point top left",
    "tblsel cell": "table select cell",
    "tblsel celltext": "table select cell text",
    "tblsel column": "table select column",
    "tblsel colbody": "table select column body",
    "tblsel row": "table select row",
    "tblsel table": "table select table",
    "table dialog paste repl": "table dialog paste replace",
    "table dialog unify cf": "table dialog unify cell format",
    "table dialog unify tf": "table dialog unify table format",
    "tc search book": "track changes search book",
    "tc search ditamap": "track changes search DITA map",
    "tc search document": "track changes search document",
    "tc search selection": "track changes search selection",
    "tc user name": "track changes user name",
    "test printdbre": "test print database",
    "textcolpro": "text column properties",
    "textltool": "text left-to-right tool",
    "textrtool": "text right-to-left tool",
    "thinspace": "thin space",
    "toggledraw": "toggle draw",
    "toggle struct and do": "toggle structure and document",
    "toolbar base": "toolbar base",
    "toolbar hideall": "toolbar hide all",
    "toolbar showall": "toolbar show all",
    "toolwindow": "tool window",
    "trackchangedisable": "track change disable",
    "uialertstrings pref": "UI alert strings preference",
    "uncond": "unconditional",
    "varcurdate": "variable current date",
    "varcurpg": "variable current page",
    "varother": "variable other",
    "varpgcount": "variable page count",
    "varquick": "variable quick",
    "verifycontext": "verify context",
    "view api shortcut": "view API shortcut",
    "view switch": "view switch",
    "vscroll": "vertical scroll",
    "width0": "width 0",
    "width1": "width 1",
    "windowfull down": "window full page down",
    "windowfull up": "window full page up",
    "win cascade": "window cascade",
    "win tile": "window tile",
    "wrapquick": "wrap quick",
    "xchars": "special characters",
    "xrefspod": "cross references pod",
    "zoom100": "zoom 100 percent",
    "zoomin": "zoom in",
    "zoomout": "zoom out",
    "defn": "definition",
    "expr": "expression",
    "def": "definition",
    "keydef": "key definition",
}

# Word-level abbreviation expander for automated expansion of terms
# not covered by PLAIN_ENGLISH_OVERRIDES.  Applied per-word.
WORD_EXPANSIONS = {
    # Single-letter / very short abbreviations
    "l":     "left",
    "r":     "right",
    "ip":    "insertion point",
    "db":    "database",
    "fm":    "FrameMaker",
    "ui":    "user interface",
    "tc":    "track changes",
    "ob":    "object",
    "lr":    "left right",
    "ud":    "up down",
    "cw":    "clockwise",
    "ccw":   "counterclockwise",
    "rc":    "rows and columns",
    "cf":    "cell format",
    "tf":    "table format",

    # Common FrameMaker / publishing abbreviations
    "pgf":   "paragraph",
    "tbl":   "table",
    "fmt":   "format",
    "doc":   "document",
    "cond":  "condition",
    "cb":    "change bar",
    "cn":    "condition",
    "dfs":   "depth first search",
    "xref":  "cross reference",
    "var":   "variable",
    "mkr":   "marker",
    "char":  "character",
    "fn":    "footnote",
    "gfx":   "graphic",
    "dlg":   "dialog",
    "btn":   "button",
    "cmd":   "command",
    "hdr":   "header",
    "ftr":   "footer",
    "bk":    "book",
    "rect":  "rectangle",
    "attr":  "attribute",
    "attrval": "attribute value",
    "elem":  "element",
    "obj":   "object",
    "sel":   "selection",
    "str":   "structure",
    "chr":   "character",
    "pg":    "page",
    "num":   "number",
    "idx":   "index",
    "cnt":   "count",
    "len":   "length",
    "pos":   "position",
    "loc":   "location",
    "rng":   "range",
    "prev":  "previous",
    "nxt":   "next",
    "beg":   "beginning",
    "ldr":   "leader",
    "dpi":   "dots per inch",
    "pct":   "percent",
    "wd":    "width",
    "ht":    "height",
    "wt":    "weight",
    "sz":    "size",
    "bg":    "background",
    "fg":    "foreground",
    "src":   "source",
    "dst":   "destination",
    "cols":  "columns",
    "col":   "column",
    "clr":   "color",
    "sep":   "separator",
    "cfg":   "configuration",
    "ctx":   "context",
    "ref":   "reference",
    "usr":   "user",
    "sys":   "system",
    "mgr":   "manager",
    "pro":   "properties",
    "prop":  "property",
    "props": "properties",
    "val":   "value",
    "vals":  "values",
    "txt":   "text",
    "para":  "paragraph",
    "horiz": "horizontal",
    "vert":  "vertical",
    "dir":   "direction",
    "win":   "window",
    "opts":  "options",
    "opt":   "option",
    "ind":   "indicator",
    "disp":  "display",
    "comp":  "component",
    "compo": "composition",
    "desig": "designer",
    "catal": "catalog",
    "overr": "override",
    "unuse": "unused",
    "inc":   "increase",
    "dec":   "decrease",
    "del":   "delete",
    "ins":   "insert",
    "upd":   "update",
    "init":  "initialize",
    "kb":    "keyboard",
    "kern":  "kern",
    "rot":   "rotate",
    "sm":    "small",
    "med":   "medium",
    "lg":    "large",
    "eqn":   "equation",
    "hyp":   "hyphen",
    "sp":    "space",
    "ti":    "text inset",
    "cms":   "content management system",
    "dup":   "duplicate",
    "inv":   "invalid",
    "spec":  "specification",
    "xrefs": "cross references",
    "res":   "resolution",
    "csr":   "cursor",
    "charfwd": "character forward",
    "eol":   "end of line",
    "eow":   "end of word",
    "eos":   "end of sentence",
    "ele":   "element",
    "coltop":"column top",
    "updateall": "update all",
    "pref":  "preference",
    "rmvie": "resource manager view",
    "repl":  "replace",
    "nopref":"no preference",
    "chap":   "chapter",
    "dfs":   "depth first search",
}

# Known compound words that should be split.
# Sorted longest-first at runtime to prevent partial matches.
COMPOUND_SPLITS = {
    "topicsetrefs":  "topic set references",
    "structapp":     "structured application",
    "notrim":        "no trim",
    "charbwd":       "character backward",
    "charfwd":       "character forward",
    "updateall":     "update all",
    "coltop":        "column top",
    "flowstructure":  "flow structure",
    "setwidth":       "set width",
    "newformat":      "new format",
    "textsel":        "text selection",
    "designkit":      "design kit",
    "dblunderline":   "double underline",
    "lessstretch":    "less stretch",
    "morestretch":    "more stretch",
    "normalcase":     "normal case",
    "allcaps":        "all caps",
    "nostretch":      "no stretch",
    "halfwidth":      "half width",
    "numunderline":   "number underline",
    "fullwidth":      "full width",
    "seldecsize":     "selection decrease size",
    "selincsize":     "selection increase size",
    "selitalic":      "selection italic",
    "selroman":       "selection roman",
    "selplain":       "selection plain",
    "selunderline":   "selection underline",
    "pagebreak":      "page break",
    "pagelayout":     "page layout",
    "pagesetup":      "page setup",
    "pagesize":       "page size",
    "pagestatus":     "page status",
    "pageupdate":     "page update",
    "pageback":       "page back",
    "pastespecial":   "paste special",
    "contextmenu":    "context menu",
    "menubarfocus":   "menu bar focus",
    "menucomplete":   "menu complete",
    "menucustom":     "menu custom",
    "menumodify":     "menu modify",
    "menuquick":      "menu quick",
    "menureset":      "menu reset",
    "softreturn":     "soft return",
    "hardreturn":     "hard return",
    "hardhyphen":     "hard hyphen",
    "hardspace":      "hard space",
    "softhyphen":     "soft hyphen",
    "thinspace":      "thin space",
    "emspace":        "em space",
    "enspace":        "en space",
    "numspace":       "numeric space",
    "insertobject":   "insert object",
    "insertquick":    "insert quick",
    "selectall":      "select all",
    "toggledraw":     "toggle draw",
    "findnext":       "find next",
    "findprev":       "find previous",
    "joincurves":     "join curves",
    "roundrect":      "round rectangle",
}


def _expand_word(word: str) -> str:
    """Expand a single word using WORD_EXPANSIONS, case-insensitive."""
    return WORD_EXPANSIONS.get(word.lower(), word)


def _auto_expand_phrase(phrase: str) -> str:
    """Automatically expand a friendly-name phrase to plain English.

    1. Check PLAIN_ENGLISH_OVERRIDES for a full-phrase match.
    2. Otherwise, split on spaces and expand each word individually.
    """
    key = phrase.strip().lower()

    # Full-phrase override (case-insensitive lookup)
    lower_overrides = {k.lower(): v for k, v in PLAIN_ENGLISH_OVERRIDES.items()}
    if key in lower_overrides:
        return lower_overrides[key]

    # Word-level expansion
    words = phrase.strip().split()
    expanded = [_expand_word(w) for w in words]
    return " ".join(expanded)


CONVERT_OUTPUT_PATH = CSV_PATH.with_name("codename_plainenglish.txt")


def convert_friendlynames_to_plain_english():
    """Read codename_friendlyname.csv, expand friendlyname column to plain
    English, then write a two-column output file:

        "friendlyname": "plain english expansion"

    - Reads columns: count, type, codename, friendlyname, review
    - Adds a plain-English expansion of the friendlyname column
    - Outputs only the last two columns (friendlyname, plain_english)
    - Wraps both in quotes separated by ": "
    """
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found. Run extract mode first.")
        return

    rows_out: list[str] = []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            friendly = row["friendlyname"].strip()
            if not friendly:
                continue
            plain = _auto_expand_phrase(friendly)
            rows_out.append(f'"{friendly}": "{plain}"')

    # Write output
    with open(CONVERT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(rows_out))
        f.write("\n")

    print(f"Wrote {len(rows_out)} entries to {CONVERT_OUTPUT_PATH}")
    print(f"Format: \"friendlyname\": \"plain english\"")
    print(f"\nSample (first 5):")
    for line in rows_out[:5]:
        print(f"  {line}")

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Extract FDK codenames and generate friendly name mappings"
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Apply approved CSV mappings to a_chunks.json "
             "(adds 'apifriendlynames' field)"
    )
    parser.add_argument(
        "--convert", action="store_true",
        help="Convert friendlyname column to plain English "
             "(writes codename_plainenglish.txt)"
    )
    args = parser.parse_args()

    if args.convert:
        convert_friendlynames_to_plain_english()
        return

    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")

    if args.apply:
        if not CSV_PATH.exists():
            print(f"ERROR: {CSV_PATH} not found. Run without --apply first.")
            return

        # Backup before mutating
        backup_path = CHUNKS_PATH.with_suffix(".json.bak")
        shutil.copy2(CHUNKS_PATH, backup_path)
        print(f"Backup saved to {backup_path}")

        chunks = apply_csv(chunks)

        with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        print(f"Saved enriched chunks to {CHUNKS_PATH}")

    else:
        codenames = extract_codenames(chunks)
        write_csv(codenames)


if __name__ == "__main__":
    main()
