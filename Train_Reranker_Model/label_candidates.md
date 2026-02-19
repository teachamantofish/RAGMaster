# Exp 33 - Label Candidates for Excluded Filter Queries

For each query, review the candidates and write the best 1-3 chunk IDs
into the `positive_ids` field. Then we'll update retriever_eval_queries.json.

---

## 1. q001: How can I find a table title in a MIF file?

**Filters:**
  - Filter 1: fields=['chunk_summary', 'content', 'heading'], contains_any=['table title', 'TblTitle'], contains_all=[], max_matches=20
  - Filter 2: fields=['content', 'chunk_summary'], contains_any=[], contains_all=['TblTitlePosition', 'Constants.FV_TBL'], max_matches=10

**Top 15 reranker-scored candidates** (filter match marked with [F]):

| Rank | Chunk ID | Score | Label | [F] | Heading | Content Preview |
|------|----------|-------|-------|-----|---------|-----------------|
| 1 | `h4_c897d9cd_tab` | 2.44 | 1 | [F] | Table title | MIF Document Statements > Tables > Tbl statement > Table title The `TblTitle` statement defines a table’s title using on... |
| 2 | `h3_fb1188a6` | -0.37 | 0 |  | Tbls statement | MIF Document Statements > Tables > Tbls statement The `Tbls` statement declares all tables in a MIF document, serving as... |
| 3 | `h3_5f801b36_exa` | -0.95 | 1 | [F] | Creating a table format | ```mif <TblFormat <TblTag `Coffee Table'> # Every table must have at least one TblColumn # statement. <TblColumn <TblCol... |
| 4 | `h4_c4bcbc6a_tab` | -2.21 | 0 |  | Miscellaneous properties | <!-- Data Table --> MIF object,Description <TblLocked (boolean)>,Yes means the table is part of a text inset that obtain... |
| 5 | `h2_faa50981_tab` | -2.56 | 1 | [F] | Constants | These constants define table formatting behaviors in MIF, categorizing layout options like title placement (above/below/... |
| 6 | `h3_9b751453_exa` | -2.69 | 0 |  | Creating a table instance | ```mif <Tbl <TblID…> # A unique ID for the table <TblFormat…> # The table format <TblNumColumns…> # Number of columns in... |
| 7 | `h2_b276d889` | -3.17 | 0 |  | Tables | MIF Document Statements > Tables Table formats, rulings, and instances in MIF are centrally managed: `TblFormat` defines... |
| 8 | `h4_a6c4de01_tab` | -3.28 | 1 | [F] | Basic properties | <!-- Data Table --> MIF object,Description <TblFormat, <TblTag (tagstring)>,Table format tag name <TblLIndent (dimension... |
| 9 | `h4_ca5d3d4e_tab` | -3.29 | 0 |  | Acrobat preferences | <!-- Data Table --> MIF object,Description <DAcrobatBookmarksIncludeTagNames (boolean)>,Yes specifies that each Acrobat ... |
| 10 | `h3_a8bad948` | -3.38 | 0 |  | Adding a Table Catalog | You can store table formats in a Table Catalog by using a `TblCatalog` statement. A document can have only one `TblCatal... |
| 11 | `h3_5c57bb4f` | -3.60 | 1 | [F] | Find | Performs the same actions as using the Find dialog box to search a document for text or other types of content. The prop... |
| 12 | `h4_5d7b9ef6` | -4.35 | 0 |  | PDF Document Info | For versions 6.0 and later, FrameMaker stores PDF File Info in the document file. FrameMaker automatically supplies valu... |
| 13 | `h2_ad11e077` | -5.19 | 0 |  | How FrameMaker identifies MIF files | MIF overview > How FrameMaker identifies MIF files FrameMaker identifies MIF files by the presence of a `MIFFile` or `Bo... |
| 14 | `h3_843d74fd` | -5.36 | 0 |  | Tables inherit properties differently | Using MIF Statements > Creating and applying character formats > Tables inherit properties differently Tables don’t inhe... |
| 15 | `h3_f70ef64a` | -5.42 | 0 |  | Creating and formatting tables | You can create tables in FrameMaker documents, edit them, and apply table formats to them. Tables can have heading rows,... |

**Filter matches NOT in top 15** (12 chunks):

| # | Chunk ID | Heading | Content Preview |
|---|----------|---------|-----------------|
| 1 | `h3_ccc12af8_tab` | Command object properties | These properties control menu item behavior in FrameMaker’s command system. CanHaveCheckMark and CheckMarkIsOn manage vi... |
| 2 | `h2_092bd847_tab` | Constants | These constants define integer flags for validating and referencing structured content in a publishing system. They cate... |
| 3 | `h2_32903cab_tab` | Constants | These constants define structured properties for tables and rulings in a document layout system. They enable programmati... |
| 4 | `h3_eb897618_tab` | Tbl object properties | The Tbl object properties define structural and formatting behaviors of tables in FrameMaker. Properties like ContentHei... |
| 5 | `h3_19c7aa03_tab` | Tbl object properties | The Tbl object properties define visual and structural attributes of tables, including colors, fill patterns, element as... |
| 6 | `h3_fa0e3f71_tab` | Tbl object properties | These properties define visual and structural formatting of a table, including borders, selection ranges, and alternatin... |
| 7 | `h3_2266dadc_tab` | TblFmt object properties | The TblFmt properties define table layout and positioning: alignment (left/center/right), vertical placement (page/colum... |
| 8 | `h3_fcc1c11d_tab` | TblFmt object properties | The TblFmt properties define table structure and behavior: title placement, numbering direction, initial row/column coun... |
| 9 | `h4_057b4bb4_tab` | GetText bit flags | The GetText bit flags define which document elements return text content when queried. Only specific structural elements... |
| 10 | `h3_e6b0ac82` | MakeTblSelection | Selects a range of cells in a table. To select an entire table, including the table title, set the topRow parameter to C... |
| ... | (2 more) | | |

**YOUR PICK (write 1-3 chunk IDs):**
```
positive_ids: [h3_eb897618_tab, h3_fcc1c11d_tab]
```

---

## 2. q002: How can I write a JSX script that saves an FM file as MIF?

**Filters:**
  - Filter 1: fields=['content', 'chunk_summary'], contains_any=['Save as MIF', 'GetSaveDefaultParams'], contains_all=[], max_matches=25
  - Filter 2: fields=['content', 'chunk_summary'], contains_any=['SimpleSave', 'saveAsName'], contains_all=[], max_matches=15

**Top 15 reranker-scored candidates** (filter match marked with [F]):

| Rank | Chunk ID | Score | Label | [F] | Heading | Content Preview |
|------|----------|-------|-------|-----|---------|-----------------|
| 1 | `h2_ac622f8e_tab` | -6.06 | 0 |  | Naming Differences Between JSX Scripts and FDK | JSX scripts simplify FDK naming by stripping prefixes and suffixes for cleaner, intuitive access. FDK’s verbose identifi... |
| 2 | `h2_6192bc4e` | -6.52 | 0 |  | Basic JSX script | This is a very basic script that creates some prompts and captures simple user input. It is a good script to test whethe... |
| 3 | `h2_f0498545` | -6.54 | 0 |  | MIFFile statement | The `MIFFile` statement identifies the file as a MIF file. The `MIFFile` statement is required and must be the first lin... |
| 4 | `h2_8f8b3c68_tab` | -7.03 | 0 |  | Data Type Mapping | Differences Between JSX Scripts and the Framemaker Developer Kit (FDK) > Data Type Mapping JSX scripts abstract FDK’s lo... |
| 5 | `h2_2e3833d7` | -7.20 | 0 |  | Add Menus and Commands | You can add custom menus with custom commands and can implement your own handlers for commands in a similar way as the F... |
| 6 | `h2_d03606f7` | -7.26 | 0 |  | MIF file layout | MIF Document Statements > MIF file layout FrameMaker writes MIF files in a strict structural order, ensuring consistency... |
| 7 | `h2_5c088240` | -7.48 | 0 |  | Register a script for the notification | The following registers this script for the `FA_Note_PostOpenDoc` notification. This notification is triggered just afte... |
| 8 | `h3_4837b874` | -7.55 | 0 |  | Device-independent pathnames | Several MIF statements require pathnames as values. You should supply a device-independent pathname so that files can ea... |
| 9 | `h2_c911307e` | -7.56 | 0 |  | Global Methods: FDK vs JSX | Not every method is accessible through a specific object. There are some methods that are not called through any objects... |
| 10 | `h3_6ddc74f3_exa` | -7.86 | 0 |  | Error Handling and Completion | ```jsx // Handle missing document or condition format else { alert("No active document found or the active document does... |
| 11 | `h3_e3ea2f94` | -7.86 | 0 |  | Editing the MIF file | Using MIF Statements > Including template files > Editing the MIF file Edit the MIF file to isolate formatting and layou... |
| 12 | `h2_ad11e077` | -7.90 | 0 |  | How FrameMaker identifies MIF files | MIF overview > How FrameMaker identifies MIF files FrameMaker identifies MIF files by the presence of a `MIFFile` or `Bo... |
| 13 | `h1_7bcc29bc` | -7.90 | 0 |  | MIF Document Statements | Most MIF statements are listed in the order that they appear in a MIF file, as described in the following section. |
| 14 | `h3_b4588fd0` | -8.17 | 0 |  | Creating a double-sided custom layout | If you import a two-sided document, you might need to specify different page layouts for right and left pages. For examp... |
| 15 | `h2_2c9e82c5` | -8.20 | 0 |  | Working with MIF files | Using MIF Statements > Working with MIF files MIF files offer a human-readable, ASCII version of FrameMaker documents, e... |

**Filter matches NOT in top 15** (9 chunks):

| # | Chunk ID | Heading | Content Preview |
|---|----------|---------|-----------------|
| 1 | `h3_ecbce3f2` | Save | The Save() method saves a book. The method allows you to script the way FrameMaker saves the file and to specify respons... |
| 2 | `h3_2346d5a8` | Save | Function Summary > Doc > Save Saves a document to a specified path with customizable save behaviors and error handling. ... |
| 3 | `h3_c12daad3` | Book methods | MIF Object Reference > Book > Book methods These methods manage book lifecycle and structure: create, import, save, and ... |
| 4 | `h2_472abbac_tab` | Constants | These constants define file save behaviors and formats in a versioned application system. They distinguish between binar... |
| 5 | `h3_3e1957a3` | Doc methods | AddNewBuildExpr, AddText, CenterOnText, Clear, ClearAllChangebars, Close, Compare, Copy, Cut, DeleteBuildExpr, DeleteTex... |
| 6 | `h3_dcda5882` | SimpleSave | The SimpleSave() method saves a book. If you set the interactive parameter to False and you specify the book's current n... |
| 7 | `h3_5ded9fb6` | SimpleSave (error codes) | On failure, the method sets FA\_errno, to one of the following values: <!-- Data Table --> Error returned to FA\_errno,R... |
| 8 | `h3_b64696d3` | SimpleSave | Saves a document or book. If you set the interactive parameter to False and specify the document or book's current name ... |
| 9 | `h3_518ba146_tab` | SimpleSave | SimpleSave returns specific error codes when saving fails, mapping low-level file issues (e.g., permissions, locks, inva... |

**YOUR PICK (write 1-3 chunk IDs):**
```
positive_ids: [h2_be9735b6]
```

---

## 3. q003: How can I write a JSX script to search for and replace text in all open files?

**Filters:**
  - Filter 1: fields=['chunk_summary', 'content'], contains_any=['FirstOpenDoc', 'NextOpenDocInSession'], contains_all=[], max_matches=25
  - Filter 2: fields=['content', 'chunk_summary'], contains_any=['Find()', 'findParams', 'FS_FindWrap'], contains_all=[], max_matches=15

**Top 15 reranker-scored candidates** (filter match marked with [F]):

| Rank | Chunk ID | Score | Label | [F] | Heading | Content Preview |
|------|----------|-------|-------|-----|---------|-----------------|
| 1 | `h2_06748218_exa` | -6.07 | 0 |  | Get paragraph text | ```jsx function getText (textObj, doc) { // Gets the text from the text object or text range var textItems, text, i; // ... |
| 2 | `h3_668ed11a_exa` | -6.25 | 0 |  | Paste | ```jsx It is illegal to specify Constants.FF_REPLACE_CELLS (0x0020)| Constants.FF_INSERT_BELOW_RIGHT (0x0008). ``` |
| 3 | `h2_ead2c852_exa` | -6.43 | 0 |  | Retrieve text over a network | ```jsx var doc = app.ActiveDoc; // Active document if (doc.ObjectValid()) { alert("Contacting www.test.com/resources/es_... |
| 4 | `h2_6192bc4e` | -6.49 | 0 |  | Basic JSX script | This is a very basic script that creates some prompts and captures simple user input. It is a good script to test whethe... |
| 5 | `h2_5c088240` | -6.57 | 0 |  | Register a script for the notification | The following registers this script for the `FA_Note_PostOpenDoc` notification. This notification is triggered just afte... |
| 6 | `h3_6ddc74f3_exa` | -7.14 | 0 |  | Error Handling and Completion | ```jsx // Handle missing document or condition format else { alert("No active document found or the active document does... |
| 7 | `h2_ac622f8e_tab` | -7.18 | 0 |  | Naming Differences Between JSX Scripts and FDK | JSX scripts simplify FDK naming by stripping prefixes and suffixes for cleaner, intuitive access. FDK’s verbose identifi... |
| 8 | `h2_8f8b3c68_tab` | -7.40 | 0 |  | Data Type Mapping | Differences Between JSX Scripts and the Framemaker Developer Kit (FDK) > Data Type Mapping JSX scripts abstract FDK’s lo... |
| 9 | `h2_2fddfa5d_exa` | -7.47 | 1 | [F] | Get all open documents | ```jsx // Get first document from FrameMaker's internal document stack (unordered) var doc = app.FirstOpenDoc; var openD... |
| 10 | `h2_cdb1ca35_exa` | -7.69 | 0 |  | Insert and replace text | ```jsx var doc = app.ActiveDoc; // Get the active document from the FrameMaker session if(doc.ObjectValid() == true) { /... |
| 11 | `h2_2e3833d7` | -7.73 | 0 |  | Add Menus and Commands | You can add custom menus with custom commands and can implement your own handlers for commands in a similar way as the F... |
| 12 | `h2_a9464f08` | -7.90 | 0 |  | Insert and replace text | This script demonstrates advanced text manipulation techniques in FrameMaker by programmatically inserting text at speci... |
| 13 | `h3_f2d29695_tab` | -8.38 | 1 | [F] | Find | Function Summary > Doc > Find The Find function locates text starting from a specified text location using a property li... |
| 14 | `h2_c911307e` | -8.46 | 0 |  | Global Methods: FDK vs JSX | Not every method is accessible through a specific object. There are some methods that are not called through any objects... |
| 15 | `h1_ce66bbd2` | -8.68 | 0 |  | JSX example scripts | ExtendScript (JSX) is similar to JavaScript. You can easily develop ExtendScript for any of the applications in FrameMak... |

**Filter matches NOT in top 15** (11 chunks):

| # | Chunk ID | Heading | Content Preview |
|---|----------|---------|-----------------|
| 1 | `h2_7880444b` | Get all open documents | This script demonstrates session-level document navigation by iterating through all open documents in FrameMaker and rep... |
| 2 | `h2_ef785cf7_exa` | Verify hypertext marker links | ```jsx // Find open document or book by file path with flexible matching options function getOpenFile(path, considerSubs... |
| 3 | `h2_3629a7b6_tab` | Constants | These constants define numeric identifiers for key document elements and states in FrameMaker’s API, enabling programmat... |
| 4 | `h2_8ca098ec_tab` | Constants | These constants define configuration and font-related properties for document processing, grouping actionable flags (e.g... |
| 5 | `h3_86a2a046_tab` | Doc object properties | The Doc object properties define core document behaviors and structural settings in MainFlow. They control flow hierarch... |
| 6 | `h3_4bce7caf_tab` | app object properties | The app object properties define core session-wide state and configuration for FrameMaker, including active documents, v... |
| 7 | `h2_203e66d5` | Creating a graphics utilities palette/menu | This script demonstrates comprehensive dialog-based user interface design by creating a sophisticated graphic utilities ... |
| 8 | `h2_3051629a_tab` | Constants | These constants define search, spell-check, and output parameters for document processing. They enable precise control o... |
| 9 | `h3_5c57bb4f` | Find | Performs the same actions as using the Find dialog box to search a document for text or other types of content. The prop... |
| 10 | `h3_d203176f_tab` | Find | Function Summary > Doc > Find The Find function returns specific error codes to diagnose failed searches. Each code maps... |
| ... | (1 more) | | |

**YOUR PICK (write 1-3 chunk IDs):**
```
positive_ids: [h2_a9464f08, h2_cdb1ca35_exa]
```

---

## 4. q004: How can I set the max width of all tables with JSX in a FrameMaker file?

**Filters:**
  - Filter 1: fields=['chunk_summary', 'content'], contains_any=['TblWidth', 'max table width', 'SetTblFmt'], contains_all=[], max_matches=20

**Top 15 reranker-scored candidates** (filter match marked with [F]):

| Rank | Chunk ID | Score | Label | [F] | Heading | Content Preview |
|------|----------|-------|-------|-----|---------|-----------------|
| 1 | `h2_43d20e69` | 4.22 | 0 |  | Changing Table Properties | The following script changes the left indentation of all the tables in a FrameMaker body page by 1 inch and the width of... |
| 2 | `h4_f820735b_tab` | -3.18 | 1 | [F] | Column properties | <!-- Data Table --> MIF object,Description <TblWidth (dimension)>,"Not generated by FrameMaker, but can be used by filte... |
| 3 | `h2_27595b92` | -5.03 | 0 |  | Table selection and sizing | This script demonstrates comprehensive table manipulation techniques in FrameMaker, including table identification, cell... |
| 4 | `h3_478cba26` | -5.24 | 0 |  | Determining table width | When FrameMaker writes MIF files, it uses `TblColumnWidth` in the `Tbl` statement to specify column width. However, filt... |
| 5 | `h4_563ea6be` | -5.71 | 0 |  | Usage | The table column statements specify the actual width of the table instance columns. They override the column widths spec... |
| 6 | `h4_cf736e9a` | -6.63 | 0 |  | Table columns | MIF Document Statements > Tables > Tbl statement > Table columns Defines table column structure by specifying total colu... |
| 7 | `h3_627997b4` | -6.78 | 0 |  | Creating a table instance | All table instances in a document are contained in a `Tbls` statement. The `Tbls` statement contains a list of `Tbl` sta... |
| 8 | `h2_2e3833d7` | -6.87 | 0 |  | Add Menus and Commands | You can add custom menus with custom commands and can implement your own handlers for commands in a similar way as the F... |
| 9 | `h3_f70ef64a` | -7.14 | 0 |  | Creating and formatting tables | You can create tables in FrameMaker documents, edit them, and apply table formats to them. Tables can have heading rows,... |
| 10 | `h3_6122a9d9` | -7.24 | 0 |  | TextRect statement | The `TextRect` statement defines a text frame. It can appear at the top level or in a `Page` or `Frame` statement. <!-- ... |
| 11 | `h1_ce66bbd2` | -7.65 | 0 |  | JSX example scripts | ExtendScript (JSX) is similar to JavaScript. You can easily develop ExtendScript for any of the applications in FrameMak... |
| 12 | `h2_c911307e` | -7.83 | 0 |  | Global Methods: FDK vs JSX | Not every method is accessible through a specific object. There are some methods that are not called through any objects... |
| 13 | `h2_48f243f2` | -7.91 | 0 |  | TblFmt | The method uses an FO\_TblFmt object to represent each table format in a document. |
| 14 | `h3_cb8abda0` | -8.05 | 0 |  | Applying a table format | You can apply a table format from the Table Catalog or you can define a table format locally. To apply a table format fr... |
| 15 | `h3_629f4365` | -8.10 | 0 |  | Creating a table format | A table format includes the following properties: * The properties specified by the Table Designer * These include the r... |

**Filter matches NOT in top 15** (4 chunks):

| # | Chunk ID | Heading | Content Preview |
|---|----------|---------|-----------------|
| 1 | `h2_96cc0c95_exa` | Table selection and sizing | ```jsx var doc = app.ActiveDoc; // Get active document from FrameMaker session if(doc.ObjectValid() == true) { // Valida... |
| 2 | `h2_32903cab_tab` | Constants | These constants define structured properties for tables and rulings in a document layout system. They enable programmati... |
| 3 | `h3_c157b3a6_tab` | Tbl object properties | The Tbl object properties define structural and visual formatting rules for tables, controlling row/column counts, dimen... |
| 4 | `h4_29e581d4` | Calculating proportional-width columns | MIF uses this formula to calculate the width of proportional-width columns: $$ \frac{n}{PTotal} \times PWidth $$ The arg... |

**YOUR PICK (write 1-3 chunk IDs):**
```
positive_ids: [h2_27595b92, h2_96cc0c95_exa]
```

---

## 5. q005: How can I use JSX to specify column widths in all tables in an open document?

**Filters:**
  - Filter 1: fields=['chunk_summary', 'content'], contains_any=['TblColumnAlignment', 'column widths', 'SetTblColWidth'], contains_all=[], max_matches=20

**Top 15 reranker-scored candidates** (filter match marked with [F]):

| Rank | Chunk ID | Score | Label | [F] | Heading | Content Preview |
|------|----------|-------|-------|-----|---------|-----------------|
| 1 | `h2_43d20e69` | 1.73 | 0 |  | Changing Table Properties | The following script changes the left indentation of all the tables in a FrameMaker body page by 1 inch and the width of... |
| 2 | `h4_563ea6be` | -0.37 | 1 | [F] | Usage | The table column statements specify the actual width of the table instance columns. They override the column widths spec... |
| 3 | `h4_cf736e9a` | -0.45 | 0 |  | Table columns | MIF Document Statements > Tables > Tbl statement > Table columns Defines table column structure by specifying total colu... |
| 4 | `h3_627997b4` | -1.81 | 1 | [F] | Creating a table instance | All table instances in a document are contained in a `Tbls` statement. The `Tbls` statement contains a list of `Tbl` sta... |
| 5 | `h3_629f4365` | -3.19 | 1 | [F] | Creating a table format | A table format includes the following properties: * The properties specified by the Table Designer * These include the r... |
| 6 | `h4_f820735b_tab` | -3.26 | 0 |  | Column properties | <!-- Data Table --> MIF object,Description <TblWidth (dimension)>,"Not generated by FrameMaker, but can be used by filte... |
| 7 | `h3_cb8abda0` | -3.58 | 1 | [F] | Applying a table format | You can apply a table format from the Table Catalog or you can define a table format locally. To apply a table format fr... |
| 8 | `h4_29e581d4` | -3.63 | 1 | [F] | Calculating proportional-width columns | MIF uses this formula to calculate the width of proportional-width columns: $$ \frac{n}{PTotal} \times PWidth $$ The arg... |
| 9 | `h3_89405a43` | -3.73 | 0 |  | AddCols | Adds columns to a table. The method returns FE\_Success on success. Returns: int Syntax: AddCols(refColNum, direction, n... |
| 10 | `h2_32903cab_tab` | -4.01 | 1 | [F] | Constants | These constants define structured properties for tables and rulings in a document layout system. They enable programmati... |
| 11 | `h2_48f243f2` | -4.79 | 0 |  | TblFmt | The method uses an FO\_TblFmt object to represent each table format in a document. |
| 12 | `h3_9b751453_exa` | -5.05 | 0 |  | Creating a table instance | ```mif <Tbl <TblID…> # A unique ID for the table <TblFormat…> # The table format <TblNumColumns…> # Number of columns in... |
| 13 | `h3_478cba26` | -5.82 | 1 | [F] | Determining table width | When FrameMaker writes MIF files, it uses `TblColumnWidth` in the `Tbl` statement to specify column width. However, filt... |
| 14 | `h2_27595b92` | -5.84 | 1 | [F] | Table selection and sizing | This script demonstrates comprehensive table manipulation techniques in FrameMaker, including table identification, cell... |
| 15 | `h3_fb1188a6` | -6.04 | 0 |  | Tbls statement | MIF Document Statements > Tables > Tbls statement The `Tbls` statement declares all tables in a MIF document, serving as... |

**Filter matches NOT in top 15** (3 chunks):

| # | Chunk ID | Heading | Content Preview |
|---|----------|---------|-----------------|
| 1 | `h2_96cc0c95_exa` | Table selection and sizing | ```jsx var doc = app.ActiveDoc; // Get active document from FrameMaker session if(doc.ObjectValid() == true) { // Valida... |
| 2 | `h3_79979696_tab` | Tbl object properties | The Tbl object properties define a table’s layout, positioning, and structure within a document. Properties like TblPlac... |
| 3 | `h3_5f801b36_exa` | Creating a table format | ```mif <TblFormat <TblTag `Coffee Table'> # Every table must have at least one TblColumn # statement. <TblColumn <TblCol... |

**YOUR PICK (write 1-3 chunk IDs):**
```
positive_ids: [h2_27595b92, h2_43d20e69]
```

---

## 6. q006: When I insert a new table, how can I specify the default table width via script?

**Filters:**
  - Filter 1: fields=['chunk_summary', 'content'], contains_any=['TblFmtOverride', 'TblWidth', 'default table width'], contains_all=[], max_matches=20

**Top 15 reranker-scored candidates** (filter match marked with [F]):

| Rank | Chunk ID | Score | Label | [F] | Heading | Content Preview |
|------|----------|-------|-------|-----|---------|-----------------|
| 1 | `h2_43d20e69` | -1.35 | 0 |  | Changing Table Properties | The following script changes the left indentation of all the tables in a FrameMaker body page by 1 inch and the width of... |
| 2 | `h4_563ea6be` | -3.12 | 0 |  | Usage | The table column statements specify the actual width of the table instance columns. They override the column widths spec... |
| 3 | `h3_b918b856` | -3.40 | 0 |  | NewAnchoredFormattedObject | Creates the following types of anchored objects: - Var - XRef - Tbl NewAnchoredFormattedObject() inserts the object at t... |
| 4 | `h3_70ccf4de` | -3.68 | 0 |  | Delete | Deletes the specified table format. See Delete under the AFrame class. |
| 5 | `h3_64fe009c_tab` | -4.01 | 0 |  | NewTable | NewTable creates a table with customizable structure using a format template. It requires explicit specification of colu... |
| 6 | `h3_e30d6de4` | -4.36 | 0 |  | Delete | Deletes the specified table. See Delete under the AFrame class for more information. Returns: int Syntax: Delete() |
| 7 | `h3_89405a43` | -4.65 | 0 |  | AddCols | Adds columns to a table. The method returns FE\_Success on success. Returns: int Syntax: AddCols(refColNum, direction, n... |
| 8 | `h3_4aa97070` | -5.43 | 0 |  | Tbl statement | The `Tbl` statement contains the contents of a table instance. It must appear in a `Tbls` statement. Each `Tbl` statemen... |
| 9 | `h4_c9b50b9e_tab` | -5.69 | 0 |  | New table properties | MIF Document Statements > Tables > TblFormat statement > New table properties Defines initial structural parameters for ... |
| 10 | `h2_27595b92` | -6.09 | 0 |  | Table selection and sizing | This script demonstrates comprehensive table manipulation techniques in FrameMaker, including table identification, cell... |
| 11 | `h4_cf736e9a` | -6.47 | 0 |  | Table columns | MIF Document Statements > Tables > Tbl statement > Table columns Defines table column structure by specifying total colu... |
| 12 | `h4_4bc567e5_tab` | -6.58 | 0 |  | Basic properties | <!-- Data Table --> |
| 13 | `h4_d8b20654_tab` | -6.58 | 0 |  | PDF properties | <!-- Data Table --> |
| 14 | `h3_f47a054b` | -6.60 | 0 |  | Creating default paragraph formats for new tables | You can use the `TblFormat` and `TblColumn` statements to define default paragraph formats for the columns in new tables... |
| 15 | `h3_775c1fd8` | -6.70 | 0 |  | push | Returns the new length of the array. Returns: number Syntax: push(value) <!-- Data Table --> Parameter name,Data Type,Op... |

**Filter matches NOT in top 15** (5 chunks):

| # | Chunk ID | Heading | Content Preview |
|---|----------|---------|-----------------|
| 1 | `h2_96cc0c95_exa` | Table selection and sizing | ```jsx var doc = app.ActiveDoc; // Get active document from FrameMaker session if(doc.ObjectValid() == true) { // Valida... |
| 2 | `h2_32903cab_tab` | Constants | These constants define structured properties for tables and rulings in a document layout system. They enable programmati... |
| 3 | `h3_c157b3a6_tab` | Tbl object properties | The Tbl object properties define structural and visual formatting rules for tables, controlling row/column counts, dimen... |
| 4 | `h4_f820735b_tab` | Column properties | <!-- Data Table --> MIF object,Description <TblWidth (dimension)>,"Not generated by FrameMaker, but can be used by filte... |
| 5 | `h4_29e581d4` | Calculating proportional-width columns | MIF uses this formula to calculate the width of proportional-width columns: $$ \frac{n}{PTotal} \times PWidth $$ The arg... |

**YOUR PICK (write 1-3 chunk IDs):**
```
positive_ids: [h3_fcc1c11d_tab, h3_d1bb5751]
```

---

## 7. q007: What is the safest way to delete a FrameMaker reference page via JSX scripting?

**Filters:**
  - Filter 1: fields=['chunk_summary', 'content', 'heading'], contains_any=['DeleteMasterPage', 'reference page', 'doc.DeletePage'], contains_all=[], max_matches=20

**Top 15 reranker-scored candidates** (filter match marked with [F]):

| Rank | Chunk ID | Score | Label | [F] | Heading | Content Preview |
|------|----------|-------|-------|-----|---------|-----------------|
| 1 | `h3_5f87d866` | -2.14 | 1 | [F] | Delete | Deletes a reference page. See Delete under the AFrame class for more information. Returns: int Syntax: Delete() |
| 2 | `h2_d1563e82` | -3.19 | 0 |  | Creating cross-references | In a FrameMaker document, you can create cross-references that are automatically updated. A cross-reference can refer to... |
| 3 | `h2_591b6223` | -3.19 | 0 |  | Delete a paragraph format | This script demonstrates format catalog management by deleting a specific paragraph format from the active document, ill... |
| 4 | `h3_ef99f6f3` | -4.40 | 0 |  | Delete | Deletes the specified text frame. See Delete under the AFrame class for more information. Returns: int Syntax: Delete() |
| 5 | `h2_cdb1ca35_exa` | -4.42 | 0 |  | Insert and replace text | ```jsx var doc = app.ActiveDoc; // Get the active document from the FrameMaker session if(doc.ObjectValid() == true) { /... |
| 6 | `h2_a9464f08` | -4.44 | 0 |  | Insert and replace text | This script demonstrates advanced text manipulation techniques in FrameMaker by programmatically inserting text at speci... |
| 7 | `h2_2e3833d7` | -4.91 | 0 |  | Add Menus and Commands | You can add custom menus with custom commands and can implement your own handlers for commands in a similar way as the F... |
| 8 | `h3_c379d146` | -5.05 | 0 |  | How FrameMaker writes cross-references | When FrameMaker writes a cross-reference, it provides the actual text that will appear at the reference point. This info... |
| 9 | `h2_06748218_exa` | -5.30 | 0 |  | Get paragraph text | ```jsx function getText (textObj, doc) { // Gets the text from the text object or text range var textItems, text, i; // ... |
| 10 | `h3_7f860344_exa` | -5.78 | 0 |  | Editing the MIF file | ```mif <Page <Unique 45155> <PageType BodyPage > <PageNum `1'> <PageSize 8.5" 11.0"> <PageOrientation Portrait > <PageAn... |
| 11 | `h2_ef9b4c3f` | -5.90 | 0 |  | Notifications | Notifications is the internal mechanism through which a script registered for a particular event is run when the event i... |
| 12 | `h2_7880444b` | -6.03 | 0 |  | Get all open documents | This script demonstrates session-level document navigation by iterating through all open documents in FrameMaker and rep... |
| 13 | `h3_f30c00eb` | -6.04 | 0 |  | Tips | The following hints may help you minimize the MIF statements for paragraph formats: * If possible, use the formats in th... |
| 14 | `h3_f7142515` | -6.06 | 0 |  | Inserting the reference source marker | To mark the location of the reference source, insert a `Marker` statement at the beginning of the reference source. The ... |
| 15 | `h2_dd4bfe74` | -6.15 | 0 |  | Configure attribute displays | This script demonstrates Structure View customization and keyboard shortcut automation by creating up to three configura... |

**Filter matches NOT in top 15** (19 chunks):

| # | Chunk ID | Heading | Content Preview |
|---|----------|---------|-----------------|
| 1 | `h3_706ec302_tab` | Doc object properties | The Doc object properties expose key structural and formatting attributes of a document, enabling programmatic control a... |
| 2 | `h3_6442759b_tab` | Doc object properties | These properties define how a Doc object interprets hypertext commands. They map parsed command details—like matrix dime... |
| 3 | `h3_ec9b6f3f_tab` | Doc object properties | The Doc object’s HypertextParse properties track and report hypertext command errors in FrameMaker. They capture validat... |
| 4 | `h3_0f3c0b00_tab` | Doc object properties | These properties expose key document elements for programmatic access, enabling navigation and manipulation of core comp... |
| 5 | `h3_dd58b742_tab` | Doc object properties | These properties expose the first instance of key document components—flows, paragraphs, graphics, markers, formats, and... |
| 6 | `h2_ed4345e8_tab` | RefPage | MIF Object Reference > RefPage The RefPage table defines essential layout properties for document pages, enabling naviga... |
| 7 | `h3_6186c3fe` | SimpleGenerate | The SimpleGenerate() method generates files for a book. The method performs the same operation as choosing Update Book f... |
| 8 | `h4_3fc27316` | Import properties for importing Framemaker and MIF... | Import() uses the following properties only for importing FrameMaker documents and MIF files : <!-- Data Table --> Prope... |
| 9 | `h3_a643ba9d` | SimpleImportFormats | Imports formats from a document to a document or a book. If you import formats to a book, the method imports formats to ... |
| 10 | `h3_3c4a4c55_tab` | SimpleImportFormats | This table maps binary flags to document format import options, enabling precise control over what elements are imported... |
| ... | (9 more) | | |

**YOUR PICK (write 1-3 chunk IDs):**
```
positive_ids: [h3_54614d1a, h3_5f87d866]
```

---

## 8. q008: What is the safest way to edit a master page via JSX scripting?

**Filters:**
  - Filter 1: fields=['chunk_summary', 'content', 'heading'], contains_any=['master page', 'GetMasterPage', 'doc.MasterPages', 'ApplyMasterPage', 'UpdateMasterPage'], contains_all=[], max_matches=40

**Top 15 reranker-scored candidates** (filter match marked with [F]):

| Rank | Chunk ID | Score | Label | [F] | Heading | Content Preview |
|------|----------|-------|-------|-----|---------|-----------------|
| 1 | `h3_2759e806` | -4.27 | 1 | [F] | Delete | Deletes a master page. See Delete under the AFrame class for more information. Returns: int Syntax: Delete() |
| 2 | `h3_148cb4b7` | -4.71 | 1 | [F] | GetNamedMasterPage | Function Summary > Doc > GetNamedMasterPage Retrieves a specific MasterPage by name from the document. This function is ... |
| 3 | `h3_f97e9c06` | -4.77 | 1 | [F] | ApplyPageLayout | Function Summary > MasterPage > ApplyPageLayout Applies the layout from a source master page to another page object—Body... |
| 4 | `h3_25217b8e` | -4.80 | 1 | [F] | SetProps | Function Summary > MasterPage > SetProps Sets master page properties via a PropVal list, applying configuration uniforml... |
| 5 | `h4_1d322d63` | -5.25 | 1 | [F] | To create the master page | To create a master page layout, use the `Page` statement to create the page and use the `TextRect` statement to create t... |
| 6 | `h2_5cc7fec6` | -5.59 | 0 |  | Adding Text and Enabling Change Bars | JSX example scripts > Adding Text and Enabling Change Bars The script inserts “Hello” at the start of the first paragrap... |
| 7 | `h2_8f8b3c68_tab` | -5.82 | 0 |  | Data Type Mapping | Differences Between JSX Scripts and the Framemaker Developer Kit (FDK) > Data Type Mapping JSX scripts abstract FDK’s lo... |
| 8 | `h3_a759c372` | -6.08 | 1 | [F] | Inserting variables | To insert a user variable or a system variable in text, use the `Variable` statement. The following example inserts the ... |
| 9 | `h3_21049a1f` | -6.14 | 1 | [F] | Using the default layout | If you don't need to control the page layout of a document, you can use the default page layout by putting all of the do... |
| 10 | `h4_d496da3d` | -6.23 | 1 | [F] | To create an empty body page | To create the body page, use the `Page` statement. Then use the `TextRect` statement to create a text frame with dimensi... |
| 11 | `h4_c2e44eed` | -6.33 | 1 | [F] | Creating a simple page layout | If you want some control of the page layout but do not want to create master pages, you can use the `Document` substatem... |
| 12 | `h2_6192bc4e` | -6.41 | 0 |  | Basic JSX script | This is a very basic script that creates some prompts and captures simple user input. It is a good script to test whethe... |
| 13 | `h4_35683d09` | -6.63 | 1 | [F] | To create the text flow for the master page | The text flow for the master page is not contained in the `Page` statement; instead, it is contained in a `TextFlow` sta... |
| 14 | `h3_d8164d52` | -6.64 | 1 | [F] | NewNamedMasterPage | Function Summary > Doc > NewNamedMasterPage Creates a named Master Page within the document, assigning it a unique ident... |
| 15 | `h2_ac622f8e_tab` | -7.04 | 0 |  | Naming Differences Between JSX Scripts and FDK | JSX scripts simplify FDK naming by stripping prefixes and suffixes for cleaner, intuitive access. FDK’s verbose identifi... |

**Filter matches NOT in top 15** (25 chunks):

| # | Chunk ID | Heading | Content Preview |
|---|----------|---------|-----------------|
| 1 | `h2_591b6223` | Delete a paragraph format | This script demonstrates format catalog management by deleting a specific paragraph format from the active document, ill... |
| 2 | `h2_96461797_tab` | BodyPage | The BodyPage object defines properties for managing individual page characteristics in a document layout. It links to ma... |
| 3 | `h2_e6bb15e6_tab` | Constants | These constants define essential page and paragraph properties for document layout and scripting in FrameMaker. They ena... |
| 4 | `h3_706ec302_tab` | Doc object properties | The Doc object properties expose key structural and formatting attributes of a document, enabling programmatic control a... |
| 5 | `h3_86a2a046_tab` | Doc object properties | The Doc object properties define core document behaviors and structural settings in MainFlow. They control flow hierarch... |
| 6 | `h3_0f3c0b00_tab` | Doc object properties | These properties expose key document elements for programmatic access, enabling navigation and manipulation of core comp... |
| 7 | `h3_dd58b742_tab` | Doc object properties | These properties expose the first instance of key document components—flows, paragraphs, graphics, markers, formats, and... |
| 8 | `h2_6565888f_tab` | MasterPage | MIF Object Reference > MasterPage The MasterPage table defines essential layout properties for document templates. Each ... |
| 9 | `h3_872d5a12_tab` | UpdateBook | The UpdateBook function orchestrates targeted book-wide updates via configurable flags, controlling whether error logs d... |
| 10 | `h3_60f4520c` | GetProps | Retrieves the properties of the master page. See GetProps under the AFrame class for more information. Returns: PropVals... |
| ... | (15 more) | | |

**YOUR PICK (write 1-3 chunk IDs):**
```
positive_ids: [h3_701e0ac0, h4_1d322d63, h4_d496da3d]
```

---

## 9. q009: How is a master page defined in MIF and what are the beginning and ending tags?

**Filters:**
  - Filter 1: fields=['content', 'chunk_summary', 'heading'], contains_any=['<MasterPage', 'MasterPageTag', 'MasterPageType', 'MasterPageMargins', '</MasterPage', 'MasterPageLayout', 'MasterPageUsage'], contains_all=[], max_matches=100
  - Filter 2: fields=['chunk_summary', 'content'], contains_any=[], contains_all=['MasterPage', 'End'], max_matches=20

**Top 15 reranker-scored candidates** (filter match marked with [F]):

| Rank | Chunk ID | Score | Label | [F] | Heading | Content Preview |
|------|----------|-------|-------|-----|---------|-----------------|
| 1 | `h3_d726e4b7` | 3.77 | 0 |  | Creating a first master page | In addition to left and right master pages, you can create custom master page layouts that you can apply to body pages. ... |
| 2 | `h2_8a181675` | 2.20 | 0 |  | Pages | Pages in a MIF file are defined by a `Page` statement. A FrameMaker document can have four types of pages: * Body pages ... |
| 3 | `h3_701e0ac0` | -0.00 | 0 |  | Creating a single-sided custom layout | Using MIF Statements > Specifying page layout > Creating a single-sided custom layout To support custom master pages in ... |
| 4 | `h4_35683d09` | -0.49 | 0 |  | To create the text flow for the master page | The text flow for the master page is not contained in the `Page` statement; instead, it is contained in a `TextFlow` sta... |
| 5 | `h3_b4588fd0` | -0.75 | 0 |  | Creating a double-sided custom layout | If you import a two-sided document, you might need to specify different page layouts for right and left pages. For examp... |
| 6 | `h4_f73f61aa` | -1.59 | 0 |  | Usage | Master and reference page names (supplied by the `PageTag` statement) appear in the status bar of a document window. The... |
| 7 | `h2_782dea2c` | -2.01 | 0 |  | Specifying page layout | FrameMaker documents have two kinds of pages that determine the position and appearance of text in the document: body pa... |
| 8 | `h4_1d322d63` | -2.19 | 1 | [F] | To create the master page | To create a master page layout, use the `Page` statement to create the page and use the `TextRect` statement to create t... |
| 9 | `h3_21049a1f` | -2.23 | 0 |  | Using the default layout | If you don't need to control the page layout of a document, you can use the default page layout by putting all of the do... |
| 10 | `h3_56b8bdd9` | -2.65 | 0 |  | Usage | MIF Document Statements > Variables > Usage Variables in MIF are named via `VariableName` and referenced by `Variable` t... |
| 11 | `h3_d8164d52` | -3.55 | 1 | [F] | NewNamedMasterPage | Function Summary > Doc > NewNamedMasterPage Creates a named Master Page within the document, assigning it a unique ident... |
| 12 | `h3_205d3e23` | -3.63 | 0 |  | Page statement | The `Page` statement adds a new page to the document. `Page` statements must appear at the top level in the order given ... |
| 13 | `h2_6565888f_tab` | -3.94 | 0 |  | MasterPage | MIF Object Reference > MasterPage The MasterPage table defines essential layout properties for document templates. Each ... |
| 14 | `h3_25217b8e` | -4.05 | 0 |  | SetProps | Function Summary > MasterPage > SetProps Sets master page properties via a PropVal list, applying configuration uniforml... |
| 15 | `h3_148cb4b7` | -4.61 | 0 |  | GetNamedMasterPage | Function Summary > Doc > GetNamedMasterPage Retrieves a specific MasterPage by name from the document. This function is ... |

**Filter matches NOT in top 15** (4 chunks):

| # | Chunk ID | Heading | Content Preview |
|---|----------|---------|-----------------|
| 1 | `h2_4953b550_tab` | Constants | These constants define page and print layout parameters for document processing, enabling precise control over page rang... |
| 2 | `h2_73f3770d_tab` | Constants | These constants define standardized identifiers for plugin metadata and output object types in the MIF system. Plugin-re... |
| 3 | `h3_86a2a046_tab` | Doc object properties | The Doc object properties define core document behaviors and structural settings in MainFlow. They control flow hierarch... |
| 4 | `h3_71a99d86_tab` | UnanchoredFrame properties | The UnanchoredFrame properties define visual and structural attributes of floating frames in a document. Key properties ... |

**YOUR PICK (write 1-3 chunk IDs):**
```
positive_ids: [h4_35683d09]
```

---

## 10. q010: How do I detect and fix broken cross-references when generating MIF from a legacy FM book?

**Filters:**
  - Filter 1: fields=['chunk_summary', 'content'], contains_any=['Broken cross-reference', 'XRefSrcText', 'UpdateXRef'], contains_all=[], max_matches=30

**Top 15 reranker-scored candidates** (filter match marked with [F]):

| Rank | Chunk ID | Score | Label | [F] | Heading | Content Preview |
|------|----------|-------|-------|-----|---------|-----------------|
| 1 | `h3_c379d146` | 2.27 | 1 | [F] | How FrameMaker writes cross-references | When FrameMaker writes a cross-reference, it provides the actual text that will appear at the reference point. This info... |
| 2 | `h3_b24ec725` | 0.41 | 1 | [F] | Inserting the reference point | The final step in creating a cross-reference is to insert an `XRef` statement at the position in text where the cross-re... |
| 3 | `h2_d1563e82` | -0.76 | 0 |  | Creating cross-references | In a FrameMaker document, you can create cross-references that are automatically updated. A cross-reference can refer to... |
| 4 | `h3_f7142515` | -4.63 | 0 |  | Inserting the reference source marker | To mark the location of the reference source, insert a `Marker` statement at the beginning of the reference source. The ... |
| 5 | `h3_e98fdb75` | -5.11 | 0 |  | XRefFormats and XRefFormat statements | The `XRefFormats` statement defines the formats of cross-references to be used in document text flows. A MIF file can ha... |
| 6 | `h3_ee038cce` | -5.95 | 0 |  | Using active cross-references | A locked document automatically has active cross-references. An *active cross-reference* behaves like a hypertext `gotol... |
| 7 | `h2_240a890d` | -5.97 | 0 |  | Cross-references | MIF Document Statements > Cross-references FrameMaker documents use cross-references to link internal or external conten... |
| 8 | `h3_ffa3d1d6` | -6.09 | 1 | [F] | UpdateXRef | Updates the cross-references in a document. It performs the same operation as clicking Update in the Cross-Reference win... |
| 9 | `h3_990893f8` | -6.40 | 0 |  | GetNamedXRefFmt | Function Summary > Doc > GetNamedXRefFmt Retrieves a named Cross Reference Format object by its identifier, enabling con... |
| 10 | `h3_321c2e0e` | -6.87 | 0 |  | NewAnchoredFormattedXRef | Function Summary > Doc > NewAnchoredFormattedXRef Creates an anchored, formatted cross-reference tied to a specific text... |
| 11 | `h3_845648d6` | -6.91 | 0 |  | NewNamedXRefFmt | Function Summary > Doc > NewNamedXRefFmt Creates a named Cross Reference Format for consistent document referencing. Ass... |
| 12 | `h2_f0498545` | -6.99 | 0 |  | MIFFile statement | The `MIFFile` statement identifies the file as a MIF file. The `MIFFile` statement is required and must be the first lin... |
| 13 | `h4_b37400a2` | -7.10 | 0 |  | Absolute pathnames | MIF overview > MIF statement syntax > Device-independent pathnames > Absolute pathnames Absolute pathnames in MIF locate... |
| 14 | `h4_87eff52c` | -7.30 | 0 |  | Why one body page? | The method you use to create body pages is different from the method that FrameMaker uses when it writes a MIF file. Whe... |
| 15 | `h2_b06f162b` | -7.45 | 0 |  | Debugging MIF files | When FrameMaker reads a MIF file, it might detect errors such as unexpected character sequences. In UNIX and Windows ver... |

**Filter matches NOT in top 15** (12 chunks):

| # | Chunk ID | Heading | Content Preview |
|---|----------|---------|-----------------|
| 1 | `h2_f02aee80_exa` | Book operations: navigation, opening, updating | ```jsx // Robust file opening function with comprehensive error handling function OpenFile(path) { var props = GetOpenDe... |
| 2 | `h3_91223bdd_exa` | Comprehensive file opening utility | ```jsx // Comprehensive file opening function with extensive parameter configuration function OpenFile(path, ignoreError... |
| 3 | `h2_11a19fea_tab` | Constants | These constants define internal identifiers for frame variables and formatting properties used in document processing, p... |
| 4 | `h2_130c6bdf_tab` | Constants | These constants define document and UI behavior settings in a publishing system, primarily controlling formatting, visib... |
| 5 | `h2_ae46c802_tab` | Constants | These constants define discrete commands for managing DITA and FrameMaker documentation workflows—triggering actions lik... |
| 6 | `h2_7c68afe4_tab` | Constants | These constants define notification codes for key events in FrameMaker’s document lifecycle—file I/O, XML/SGML processin... |
| 7 | `h2_44dde14d_tab` | Constants | These constants define configuration flags for file operations in a document system, primarily controlling how files ope... |
| 8 | `h3_a219b05a_tab` | Doc object properties | These properties control document behavior during save and open operations. DocSaveType defines the output format (binar... |
| 9 | `h3_3e1957a3` | Doc methods | AddNewBuildExpr, AddText, CenterOnText, Clear, ClearAllChangebars, Close, Compare, Copy, Cut, DeleteBuildExpr, DeleteTex... |
| 10 | `h2_88705d5b_tab` | XRef | The XRef table defines cross-reference metadata linking source content to client-generated references. It distinguishes ... |
| ... | (2 more) | | |

**YOUR PICK (write 1-3 chunk IDs):**
```
positive_ids: [h2_be6149de_tab, h3_dab46513]
```

---

## 11. q011: Can JSX batch-update conditional text visibility across all components of a FrameMaker book?

**Filters:**
  - Filter 1: fields=['chunk_summary', 'content', 'heading'], contains_any=['ConditionalText', 'Conditional visibility', 'SetCondVisibility', 'ApplyCondSetting', 'CondFmt'], contains_all=[], max_matches=60

**Top 15 reranker-scored candidates** (filter match marked with [F]):

| Rank | Chunk ID | Score | Label | [F] | Heading | Content Preview |
|------|----------|-------|-------|-----|---------|-----------------|
| 1 | `h3_ea62e5d2` | 0.90 | 0 |  | ManageConditionalExpressions | Add, edit, or delete conditional expression tags to the current book. Applies to the options available in the Add/Edit C... |
| 2 | `h3_d3a595dc_exa` | 0.35 | 0 |  | How FrameMaker writes a conditional document | ```mif # This text flow contains the sentence as it appears in # the document body. <TextFlow <TFTag `A'> <TFAutoConnect... |
| 3 | `h3_6b1c78fd` | -0.63 | 0 |  | ApplyConditionalSettings | Apply conditional settings in the selected book based on the specified settings. Returns: Void Syntax: ApplyConditionalS... |
| 4 | `h3_c0a8189e` | -2.50 | 0 |  | Showing and hiding conditional text using Boolean ... | You can also use Boolean expressions to show or hide conditional text. Boolean condition expressions are identified usin... |
| 5 | `h3_fdf0066e` | -3.06 | 0 |  | How FrameMaker writes a conditional document | If you are converting a MIF file that was generated by FrameMaker, you need to understand how FrameMaker writes a file t... |
| 6 | `h2_babfd83a` | -3.72 | 0 |  | Conditional text | MIF Document Statements > Conditional text MIF files manage conditional text via `Condition` statements that define visi... |
| 7 | `h3_2f0d990a` | -3.78 | 0 |  | BoolCond statement | The `BoolCond` statement defines a new boolean expression, which is used to evaluate the show/hide state of conditional ... |
| 8 | `h3_6186c3fe` | -4.34 | 0 |  | SimpleGenerate | The SimpleGenerate() method generates files for a book. The method performs the same operation as choosing Update Book f... |
| 9 | `h3_6d92cb7a_tab` | -5.40 | 0 |  | UpdateBook | The UpdateBook function configures how FrameMaker handles book updates under edge conditions. It controls user notificat... |
| 10 | `h2_06748218_exa` | -5.43 | 0 |  | Get paragraph text | ```jsx function getText (textObj, doc) { // Gets the text from the text object or text range var textItems, text, i; // ... |
| 11 | `h2_3f6d5aba_tab` | -5.87 | 1 | [F] | CondFmt | The CondFmt table defines properties controlling how conditional formatting is displayed and styled in a document. It en... |
| 12 | `h3_9ad4387d` | -5.92 | 0 |  | Condition statement | The `Condition` statement defines the state of a condition tag and its condition indicators, which control how condition... |
| 13 | `h4_bbdbd7bc_tab` | -5.99 | 0 |  | Conditional text defaults | <!-- Data Table --> MIF object,Description <DShowAllConditions (boolean)>,Shows or hides all conditional text <DDisplayO... |
| 14 | `h3_0c7d6f50` | -6.10 | 0 |  | Creating and applying condition tags | In MIF, all condition tags are defined in a `ConditionCatalog` statement, which contains one or more `Condition` stateme... |
| 15 | `h2_45a13e41` | -6.10 | 0 |  | Creating conditional text | Using MIF Statements > Creating conditional text You can generate multiple document variants from one source by tagging ... |

**Filter matches NOT in top 15** (22 chunks):

| # | Chunk ID | Heading | Content Preview |
|---|----------|---------|-----------------|
| 1 | `h3_40ac72ca_exa` | Document and Condition Initialization | Report document properties > Applying conditions based on paragraph formats > Document and Condition Initialization The ... |
| 2 | `h2_011e1193_tab` | Column | MIF Object Reference > Column This column object defines structural and visibility relationships within a table. It link... |
| 3 | `h3_b63c6c73` | CondFmt methods | Delete, GetProps, ObjectValid, SetProps. |
| 4 | `h2_a42181a1_tab` | Constants | These constants define text formatting and document behavior options in a publishing system. They control line alignment... |
| 5 | `h2_3629a7b6_tab` | Constants | These constants define numeric identifiers for key document elements and states in FrameMaker’s API, enabling programmat... |
| 6 | `h2_01291df9_tab` | Constants | These constants define object types in the FrameMaker API, each mapped to a unique integer ID for programmatic identific... |
| 7 | `h3_706ec302_tab` | Doc object properties | The Doc object properties expose key structural and formatting attributes of a document, enabling programmatic control a... |
| 8 | `h3_eb26ff45_tab` | Doc object properties | These Doc object properties control fine-grained text styling and positioning: font weight, kerning (horizontal/vertical... |
| 9 | `h3_ff4ba69c_tab` | Doc object properties | The Doc object properties define typographic and formatting controls for text in MIF documents. Properties like FontFami... |
| 10 | `h3_3e1957a3` | Doc methods | AddNewBuildExpr, AddText, CenterOnText, Clear, ClearAllChangebars, Close, Compare, Copy, Cut, DeleteBuildExpr, DeleteTex... |
| ... | (12 more) | | |

**YOUR PICK (write 1-3 chunk IDs):**
```
positive_ids: [h3_0c7d6f50, h4_bbdbd7bc_tab]
```

---

## 12. q012: Which MIF attributes control side-head placement?

**Filters:**
  - Filter 1: fields=['chunk_summary', 'content', 'heading'], contains_any=['SideHead', 'SideHeadPlacement', 'RuninHead', 'MIF sidehead'], contains_all=[], max_matches=40

**Top 15 reranker-scored candidates** (filter match marked with [F]):

| Rank | Chunk ID | Score | Label | [F] | Heading | Content Preview |
|------|----------|-------|-------|-----|---------|-----------------|
| 1 | `h4_e65b0981` | -2.67 | 1 | [F] | Usage | Most MIF generators will put all document text in one `TextFlow` statement. However, if there are subsequent `TextFlow` ... |
| 2 | `h4_0f1df80f_tab` | -4.93 | 0 |  | Font name | MIF Document Statements > Character formats > PgfFont and Font statements > Font name These fields define font attribute... |
| 3 | `h3_896965c6` | -5.20 | 0 |  | AttrCondExprCatalog statement | MIF Document Statements > Filter By Attribute > AttrCondExprCatalog statement The `AttrCondExprCatalog` statement declar... |
| 4 | `h2_cfbd391a_exa` | -5.41 | 0 |  | Creating filters | Using MIF Statements > Creating filters This MIF snippet defines two attribute-based filter expressions within a catalog... |
| 5 | `h2_d03606f7` | -6.01 | 0 |  | MIF file layout | MIF Document Statements > MIF file layout FrameMaker writes MIF files in a strict structural order, ensuring consistency... |
| 6 | `h2_3311321e_tab` | -6.97 | 0 |  | Constants | These constants define control and state values for UI interactions in MIF. Positive values (1719–1731) configure visual... |
| 7 | `h4_121ba283` | -6.98 | 0 |  | Paragraph placement across text columns and side h... | MIF Document Statements > Paragraph formats > Pgf statement > Paragraph placement across text columns and side heads The... |
| 8 | `h3_8a5d4d31_tab` | -7.36 | 1 | [F] | TextFrame object properties | These properties control the visual and spatial behavior of text frames and associated arrow graphics. Side head setting... |
| 9 | `h2_7e332fd1` | -7.37 | 0 |  | Creating filters | Structured FrameMaker allows specific components in a structured document to be processed differently to generate differ... |
| 10 | `h3_db685906_tab` | -7.40 | 0 |  | Element object properties | These properties define an element’s visual behavior, structural role, and navigation within the hierarchy. AttrDisplay ... |
| 11 | `h4_bc9bd4ea_tab` | -7.44 | 1 | [F] | Pagination properties | <!-- Data Table --> MIF object,Description <PgfPlacement (keyword)>,"Vertical placement of paragraph in text column, `ke... |
| 12 | `h3_455af429` | -7.50 | 0 |  | DefAttrValues statement | MIF Document Statements > Filter By Attribute > DefAttrValues statement The `DefAttrValues` statement establishes named ... |
| 13 | `h4_f2f4ad45` | -7.61 | 0 |  | Rotated cells | Using MIF Statements > Creating and applying character formats > Adding a table anchor > Rotated cells Rotated cells in ... |
| 14 | `h2_9bec4e37_tab` | -7.71 | 0 |  | Constants | These constants define configuration behaviors for element attributes and structured document processing in MIF. They co... |
| 15 | `h2_dd830109_tab` | -7.86 | 0 |  | Constants | These constants define bit flags for text formatting attributes in MIF, each representing a unique stylistic or typograp... |

**Filter matches NOT in top 15** (14 chunks):

| # | Chunk ID | Heading | Content Preview |
|---|----------|---------|-----------------|
| 1 | `h2_33e79dd3_tab` | Constants | These constants define precise typographic and layout behaviors for text flows and table cells in MIF. They control sync... |
| 2 | `h2_40a4a5aa_tab` | Constants | These constants define layout and formatting properties for frame properties in a publishing system. They control column... |
| 3 | `h2_1abee30c_tab` | Constants | These constants define configurable properties and object relationships in FrameMaker’s document model. They control ren... |
| 4 | `h2_0e992ac0_tab` | Constants | These constants define property flags for frame and text block formatting in MIF, enabling precise control over position... |
| 5 | `h2_e625688d_tab` | Constants | These constants define paragraph and cell formatting behaviors in FV (format value) and FP (format property) namespaces.... |
| 6 | `h3_11c2edd4_tab` | FCodes object properties | These FCodes define keyboard-driven commands for structured editing in MIF, mapping hex values to specific text, table, ... |
| 7 | `h2_3e8c6534_tab` | Flow | This flow configuration table defines key layout and behavior properties for text flows in FrameMaker. It controls auto-... |
| 8 | `h3_19cc2b37_tab` | FmtChangeList object properties | The FmtChangeList properties define paragraph and character formatting overrides, enabling precise control over text app... |
| 9 | `h2_b52b0fe6_tab` | GraphicsFmt | The GraphicsFmt table defines visual and layout properties for graphical objects in a document flow. It controls appeara... |
| 10 | `h3_df66b9a0_tab` | Pgf object properties | The Pgf object properties define paragraph behavior and positioning within a document. They control placement (side, run... |
| ... | (4 more) | | |

**YOUR PICK (write 1-3 chunk IDs):**
```
positive_ids: [h4_bc9bd4ea_tab, h3_6122a9d9]
```

---

## 13. q013: How do I programmatically merge multiple FM books into one while preserving master pages?

**Filters:**
  - Filter 1: fields=['chunk_summary', 'content'], contains_any=['BookComponent', 'ImportFormats', 'MasterPage', 'Combine books'], contains_all=[], max_matches=40

**Top 15 reranker-scored candidates** (filter match marked with [F]):

| Rank | Chunk ID | Score | Label | [F] | Heading | Content Preview |
|------|----------|-------|-------|-----|---------|-----------------|
| 1 | `h3_d726e4b7` | -4.07 | 0 |  | Creating a first master page | In addition to left and right master pages, you can create custom master page layouts that you can apply to body pages. ... |
| 2 | `h3_21049a1f` | -4.68 | 0 |  | Using the default layout | If you don't need to control the page layout of a document, you can use the default page layout by putting all of the do... |
| 3 | `h4_1d322d63` | -6.25 | 0 |  | To create the master page | To create a master page layout, use the `Page` statement to create the page and use the `TextRect` statement to create t... |
| 4 | `h4_c2e44eed` | -7.08 | 0 |  | Creating a simple page layout | If you want some control of the page layout but do not want to create master pages, you can use the `Document` substatem... |
| 5 | `h3_b4588fd0` | -7.18 | 0 |  | Creating a double-sided custom layout | If you import a two-sided document, you might need to specify different page layouts for right and left pages. For examp... |
| 6 | `h4_35683d09` | -7.40 | 0 |  | To create the text flow for the master page | The text flow for the master page is not contained in the `Page` statement; instead, it is contained in a `TextFlow` sta... |
| 7 | `h3_701e0ac0` | -7.43 | 0 |  | Creating a single-sided custom layout | Using MIF Statements > Specifying page layout > Creating a single-sided custom layout To support custom master pages in ... |
| 8 | `h2_782dea2c` | -7.49 | 0 |  | Specifying page layout | FrameMaker documents have two kinds of pages that determine the position and appearance of text in the document: body pa... |
| 9 | `h4_d496da3d` | -7.55 | 0 |  | To create an empty body page | To create the body page, use the `Page` statement. Then use the `TextRect` statement to create a text frame with dimensi... |
| 10 | `h3_e96c3171` | -7.93 | 0 |  | ApplyPageLayout | Applies the layout of a page to another page. PageObject is any Page object i.e., BodyPage, MasterPage, HiddenPage. The ... |
| 11 | `h3_2b93e489` | -8.20 | 0 |  | ApplyPageLayout | Function Summary > RefPage > ApplyPageLayout Applies a layout from one page to another, using any PageObject (BodyPage, ... |
| 12 | `h3_f97e9c06` | -8.30 | 0 |  | ApplyPageLayout | Function Summary > MasterPage > ApplyPageLayout Applies the layout from a source master page to another page object—Body... |
| 13 | `h3_d8164d52` | -8.34 | 0 |  | NewNamedMasterPage | Function Summary > Doc > NewNamedMasterPage Creates a named Master Page within the document, assigning it a unique ident... |
| 14 | `h3_148cb4b7` | -8.37 | 0 |  | GetNamedMasterPage | Function Summary > Doc > GetNamedMasterPage Retrieves a specific MasterPage by name from the document. This function is ... |
| 15 | `h3_2759e806` | -8.60 | 0 |  | Delete | Deletes a master page. See Delete under the AFrame class for more information. Returns: int Syntax: Delete() |

**Filter matches NOT in top 15** (40 chunks):

| # | Chunk ID | Heading | Content Preview |
|---|----------|---------|-----------------|
| 1 | `h2_0bf04d3e` | Book operations: navigation, opening, updating | This script demonstrates comprehensive book management by navigating book components, opening files with robust error ha... |
| 2 | `h2_7f5a1ad7_exa` | Verify hypertext marker links | ```jsx // Process all open documents in a book function checkHypertextLinks_Book(book) { // Initialize book-level report... |
| 3 | `h2_96461797_tab` | BodyPage | The BodyPage object defines properties for managing individual page characteristics in a document layout. It links to ma... |
| 4 | `h3_c12daad3` | Book methods | MIF Object Reference > Book > Book methods These methods manage book lifecycle and structure: create, import, save, and ... |
| 5 | `h3_3c542fc3_tab` | BookComponent object properties | These properties define how a BookComponent is generated, typed, parented, and numbered. BookComponentIsGeneratable and ... |
| 6 | `h3_dca822b8_tab` | BookComponent object properties | These properties define metadata and traversal capabilities for book components in a structured document hierarchy. Subs... |
| 7 | `h3_d7844a1e_tab` | BookComponent object properties | These properties control section and subsection numbering in a BookComponent. SectionNumber and SectionNumText define th... |
| 8 | `h3_74a2d3f7_tab` | BookComponent object properties | These properties define a BookComponent’s identity and behavior in a FrameMaker book. BookComponentTitle names the compo... |
| 9 | `h3_1d0a36c1_tab` | BookComponent object properties | The BookComponent properties define structural and formatting behaviors within a book hierarchy. VolumeNumStyle controls... |
| 10 | `h3_5f7acaec_tab` | BookComponent object properties | These properties control footnote and volume numbering behaviors in book components. TblFnNumStyle defines table footnot... |
| ... | (30 more) | | |

**YOUR PICK (write 1-3 chunk IDs):**
```
positive_ids: [delete this question]
```

---

## 14. q014: What How can I delete all index markers?

**Filters:**
  - Filter 1: fields=['chunk_summary', 'content'], contains_any=['index marker', 'MType 2', 'Marker statement', 'MarkerType'], contains_all=[], max_matches=60

**Top 15 reranker-scored candidates** (filter match marked with [F]):

| Rank | Chunk ID | Score | Label | [F] | Heading | Content Preview |
|------|----------|-------|-------|-----|---------|-----------------|
| 1 | `h3_437c3708` | -1.04 | 0 |  | Delete | Deletes a marker type. See Delete under the AFrame class for more information. Returns: int Syntax: Delete() |
| 2 | `h3_05391b2e` | -1.48 | 0 |  | Delete | Deletes a marker. See Delete under the AFrame class for more information. Returns: int Syntax: Delete() |
| 3 | `h2_6ad963d1` | -6.29 | 1 | [F] | Creating markers | A FrameMaker document can contain markers that hold hidden text and mark locations. For example, you use markers to add ... |
| 4 | `h3_26596ca7` | -7.60 | 0 |  | DeleteAllKeyDefinitions | Function Summary > KeyCatalog > DeleteAllKeyDefinitions Deletes all key definitions in a specified key catalog, clearing... |
| 5 | `h3_38399768` | -7.79 | 0 |  | DeleteCols | Deletes columns from a table. To delete an entire table, use Delete(). The method deletes the column specified by delCol... |
| 6 | `h4_0f50550a_tab` | -8.58 | 0 |  | Miscellaneous properties | <!-- Data Table --> MIF object,Description <DMagicMarker (integer)>,Type number of the marker used to represent a delete... |
| 7 | `h3_7b8614cf` | -8.76 | 0 |  | Delete | Deletes an element. See Delete under the AFrame class for more information. Returns: int Syntax: Delete() |
| 8 | `h3_0b41e9ad` | -8.83 | 0 |  | Delete | Deletes the specified color object. See Delete under the AFrame class for more information. Returns: int Syntax: Delete(... |
| 9 | `h3_08868ba0` | -8.95 | 0 |  | pop | Removes the last element from the array. Returns: Tab Syntax: pop() |
| 10 | `h3_a4313269` | -9.04 | 0 |  | pop | Removes the last element from the array. Returns: Point Syntax: pop() |
| 11 | `h3_ad801313` | -9.21 | 0 |  | Delete | Deletes an ellipse. See Delete under the AFrame class for more information. Returns: int Syntax: Delete() |
| 12 | `h3_e30d6de4` | -9.30 | 0 |  | Delete | Deletes the specified table. See Delete under the AFrame class for more information. Returns: int Syntax: Delete() |
| 13 | `h2_c6517ccc_tab` | -9.33 | 0 |  | CMSDeleteParam | MIF Object Reference > CMSDeleteParam The CMSDeleteParam table defines two boolean flags for bulk deletion operations: d... |
| 14 | `h3_8573584f` | -9.34 | 0 |  | Delete | Deletes the specified Command object. The method does not take any arguments. Call the Delete() method directly on the o... |
| 15 | `h3_334b2013` | -9.38 | 0 |  | Delete | Deletes a format rule. See Delete under the AFrame class for more information. Returns: int Syntax: Delete() |

**Filter matches NOT in top 15** (18 chunks):

| # | Chunk ID | Heading | Content Preview |
|---|----------|---------|-----------------|
| 1 | `h2_1aafccda_exa` | Verify hypertext marker links | ```jsx // Process hypertext markers in a single document function checkHypertextLinks_Doc(doc, doingABook) { // Initiali... |
| 2 | `h2_b0f692dc_tab` | Constants | These constants define unique integer identifiers for frame, footnote, marker, and variable properties in FramerScript. ... |
| 3 | `h2_130c6bdf_tab` | Constants | These constants define document and UI behavior settings in a publishing system, primarily controlling formatting, visib... |
| 4 | `h2_3c9577d0_tab` | Constants | These constants define integer identifiers for formatting and object types within the MIF system, enabling precise type ... |
| 5 | `h3_86a2a046_tab` | Doc object properties | The Doc object properties define core document behaviors and structural settings in MainFlow. They control flow hierarch... |
| 6 | `h3_dd58b742_tab` | Doc object properties | These properties expose the first instance of key document components—flows, paragraphs, graphics, markers, formats, and... |
| 7 | `h3_3e1957a3` | Doc methods | AddNewBuildExpr, AddText, CenterOnText, Clear, ClearAllChangebars, Close, Compare, Copy, Cut, DeleteBuildExpr, DeleteTex... |
| 8 | `h3_1bf248b1_tab` | FCodes object properties | The FCodes object maps keyboard and menu commands to internal hexadecimal identifiers, enabling system-wide command reso... |
| 9 | `h2_d8a92e9f_tab` | Marker | This table defines key properties of a Marker object in FrameMaker, enabling precise identification, positioning, and cu... |
| 10 | `h2_cb343229_tab` | MarkerType | This table defines six properties for MarkerType, controlling how marker types behave in FrameMaker. It specifies visibi... |
| ... | (8 more) | | |

**YOUR PICK (write 1-3 chunk IDs):**
```
positive_ids: [h2_6ad963d1, h2_d46584ad]
```

---

## 15. q015: How can I delete all variables except for the default ones in all open files?

**Filters:**
  - Filter 1: fields=['chunk_summary', 'content'], contains_any=['DeleteUnusedVarFmt', 'GetNamedVarFmt', 'NamedVariable', 'doc.DeleteFmt'], contains_all=[], max_matches=60

**Top 15 reranker-scored candidates** (filter match marked with [F]):

| Rank | Chunk ID | Score | Label | [F] | Heading | Content Preview |
|------|----------|-------|-------|-----|---------|-----------------|
| 1 | `h3_8573584f` | -7.25 | 0 |  | Delete | Deletes the specified Command object. The method does not take any arguments. Call the Delete() method directly on the o... |
| 2 | `h3_26596ca7` | -7.67 | 0 |  | DeleteAllKeyDefinitions | Function Summary > KeyCatalog > DeleteAllKeyDefinitions Deletes all key definitions in a specified key catalog, clearing... |
| 3 | `h3_d34613f8` | -7.83 | 0 |  | pop | Removes the last element from the array. Returns: int Syntax: pop() |
| 4 | `h3_256cf52a` | -7.83 | 0 |  | pop | Removes the last element from the array. Returns: int Syntax: pop() |
| 5 | `h3_a5cef710` | -8.01 | 0 |  | pop | Removes the last element from the array. Returns: string Syntax: pop() |
| 6 | `h3_f0d0f3d1` | -8.13 | 0 |  | UpdateVariables | Function Summary > Doc > UpdateVariables Updates all document variables in one go, mimicking the manual “Update” action ... |
| 7 | `h3_32c50f13` | -8.20 | 0 |  | pop | Removes the last element from the array. Returns: Attribute Syntax: pop() |
| 8 | `h3_7b8614cf` | -8.29 | 0 |  | Delete | Deletes an element. See Delete under the AFrame class for more information. Returns: int Syntax: Delete() |
| 9 | `h3_7ffca713` | -8.36 | 0 |  | pop | Removes the last element from the array. Returns: Font Syntax: pop() |
| 10 | `h3_a759c372` | -8.37 | 0 |  | Inserting variables | To insert a user variable or a system variable in text, use the `Variable` statement. The following example inserts the ... |
| 11 | `h3_a613793c` | -8.50 | 0 |  | pop | Removes the last element from the array. Returns: AttributeEx Syntax: pop() |
| 12 | `h3_e30d6de4` | -8.65 | 0 |  | Delete | Deletes the specified table. See Delete under the AFrame class for more information. Returns: int Syntax: Delete() |
| 13 | `h3_a39b7364` | -8.73 | 0 |  | NewNamedVarFmt | Function Summary > Doc > NewNamedVarFmt Creates and returns a new Variable Format (VarFmt) with the specified name, enab... |
| 14 | `h3_eb4626a4` | -8.92 | 0 |  | VariableFormats and VariableFormat statements | MIF Document Statements > Variables > VariableFormats and VariableFormat statements The `VariableFormats` statement decl... |
| 15 | `h3_334b2013` | -8.96 | 0 |  | Delete | Deletes a format rule. See Delete under the AFrame class for more information. Returns: int Syntax: Delete() |

**Filter matches NOT in top 15** (4 chunks):

| # | Chunk ID | Heading | Content Preview |
|---|----------|---------|-----------------|
| 1 | `h2_3051629a_tab` | Constants | These constants define search, spell-check, and output parameters for document processing. They enable precise control o... |
| 2 | `h3_3e1957a3` | Doc methods | AddNewBuildExpr, AddText, CenterOnText, Clear, ClearAllChangebars, Close, Compare, Copy, Cut, DeleteBuildExpr, DeleteTex... |
| 3 | `h3_845ffe57` | GetNamedVarFmt | Function Summary > Doc > GetNamedVarFmt Retrieves a named Variable Format object by its identifier, enabling access to f... |
| 4 | `h3_618d0d50_tab` | Find | The Find table defines configuration parameters for text and marker searches in a document. It specifies search targets ... |

**YOUR PICK (write 1-3 chunk IDs):**
```
positive_ids: [h3_3eh2_42620f51_tab, h2_a97f3127_tab, h3_85ebc697_tab]
```

---

## 16. q016: How can I script a diff between two FrameMaker documents to highlight paragraph overrides?

**Filters:**
  - Filter 1: fields=['chunk_summary', 'content'], contains_any=['CompareDocuments', 'Diff', 'ParagraphOverrides', 'Highlight overrides'], contains_all=[], max_matches=40

**Top 15 reranker-scored candidates** (filter match marked with [F]):

| Rank | Chunk ID | Score | Label | [F] | Heading | Content Preview |
|------|----------|-------|-------|-----|---------|-----------------|
| 1 | `h2_cdb1ca35_exa` | -4.02 | 0 |  | Insert and replace text | ```jsx var doc = app.ActiveDoc; // Get the active document from the FrameMaker session if(doc.ObjectValid() == true) { /... |
| 2 | `h3_b4c46d4e` | -4.03 | 0 |  | Creating and applying paragraph formats | In a FrameMaker document, paragraphs have formatting properties that specify the appearance of the paragraph's text. A p... |
| 3 | `h3_8973c881` | -4.71 | 1 | [F] | Compare | Compares two documents or two books. You can OR the values shown in the following table into the flags argument. <!-- Da... |
| 4 | `h2_a9464f08` | -5.07 | 0 |  | Insert and replace text | This script demonstrates advanced text manipulation techniques in FrameMaker by programmatically inserting text at speci... |
| 5 | `h3_ef1c742e` | -5.37 | 1 | [F] | Compare | The Compare() method compares the differences between two versions of files and stores the result in a CompareRet data o... |
| 6 | `h4_402be5c1_tab` | -6.14 | 0 |  | Document view properties | <!-- Data Table --> MIF object,Description <DGridOn (boolean)>,Turns on page grid upon opening <DPageGrid (dimension)>,S... |
| 7 | `h2_5cc7fec6` | -6.30 | 0 |  | Adding Text and Enabling Change Bars | JSX example scripts > Adding Text and Enabling Change Bars The script inserts “Hello” at the start of the first paragrap... |
| 8 | `h2_fd9f0b14_exa` | -6.35 | 0 |  | Creating and applying character formats | ```mif <MIFFile 2015> # Hand generated <FontCatalog <Font <FTag `Emphasis'> <FAngle `Italic'> > # end of Font > # end of... |
| 9 | `h3_ee038cce` | -6.37 | 0 |  | Using active cross-references | A locked document automatically has active cross-references. An *active cross-reference* behaves like a hypertext `gotol... |
| 10 | `h3_784f10af` | -6.44 | 0 |  | Applying a paragraph format | To apply a format from the Paragraph Catalog to a paragraph, use the `PgfTag` statement to include the format tag name w... |
| 11 | `h2_d1563e82` | -6.45 | 0 |  | Creating cross-references | In a FrameMaker document, you can create cross-references that are automatically updated. A cross-reference can refer to... |
| 12 | `h2_0d9119b1` | -6.50 | 0 |  | Report all paragraph formats | This script demonstrates comprehensive format catalog analysis by iterating through all paragraph formats in the active ... |
| 13 | `h2_7880444b` | -6.53 | 0 |  | Get all open documents | This script demonstrates session-level document navigation by iterating through all open documents in FrameMaker and rep... |
| 14 | `h2_43d20e69` | -6.55 | 0 |  | Changing Table Properties | The following script changes the left indentation of all the tables in a FrameMaker body page by 1 inch and the width of... |
| 15 | `h2_e7e06a4f` | -6.71 | 0 |  | Report paragraphs and their formats | This script demonstrates comprehensive paragraph analysis by iterating through the main flow to extract both text conten... |

**Filter matches NOT in top 15** (38 chunks):

| # | Chunk ID | Heading | Content Preview |
|---|----------|---------|-----------------|
| 1 | `h1_818e46bb` | Differences Between JSX Scripts and the Framemaker... | Adobe FrameMaker scripts are modeled closely on the FrameMaker FDK. These scripts act as wrappers to the FDK and hide th... |
| 2 | `h2_ac622f8e_tab` | Naming Differences Between JSX Scripts and FDK | JSX scripts simplify FDK naming by stripping prefixes and suffixes for cleaner, intuitive access. FDK’s verbose identifi... |
| 3 | `h2_8f8b3c68_tab` | Data Type Mapping | Differences Between JSX Scripts and the Framemaker Developer Kit (FDK) > Data Type Mapping JSX scripts abstract FDK’s lo... |
| 4 | `h2_987aadfa` | Example: Calling Methods: FDK vs JSX | Differences Between JSX Scripts and the Framemaker Developer Kit (FDK) > Example: Calling Methods: FDK vs JSX Example 2 ... |
| 5 | `h2_2e3833d7` | Add Menus and Commands | You can add custom menus with custom commands and can implement your own handlers for commands in a similar way as the F... |
| 6 | `h2_06748218_exa` | Get paragraph text | ```jsx function getText (textObj, doc) { // Gets the text from the text object or text range var textItems, text, i; // ... |
| 7 | `h2_b879acf1` | Resizing and rotating graphics | This script demonstrates animated graphic manipulation by programmatically resizing and rotating a selected graphic thro... |
| 8 | `h2_9ed0fb9a` | Apply character formats | This script demonstrates programmatic character formatting by applying the BoldRed character format to a specific text r... |
| 9 | `h2_f6a6dd29` | Create menus and extend the FrameMaker user interf... | This script demonstrates comprehensive menu system integration by creating custom menus with commands, implementing keyb... |
| 10 | `h2_eedda505_exa` | Create menus and extend the FrameMaker user interf... | ```jsx // Define command response handler for menu item invocations function Command(cmd){ // Handle different commands ... |
| ... | (28 more) | | |

**YOUR PICK (write 1-3 chunk IDs):**
```
positive_ids: [h3_be4b9cd4]
```

---

## 17. q017: What is the procedure to bulk replace fonts in FM using ExtendScript without corrupting styles?

**Filters:**
  - Filter 1: fields=['chunk_summary', 'content'], contains_any=['ReplaceFont', 'FontCatalog', 'SetTextFmt', 'ChangeFonts'], contains_all=[], max_matches=40

**Top 15 reranker-scored candidates** (filter match marked with [F]):

| Rank | Chunk ID | Score | Label | [F] | Heading | Content Preview |
|------|----------|-------|-------|-----|---------|-----------------|
| 1 | `h3_f0c2b17e` | -4.77 | 0 |  | concat | Function Summary > Fonts > concat The `concat` method creates a new Fonts array by appending provided values—including e... |
| 2 | `h3_7ffca713` | -6.84 | 0 |  | pop | Removes the last element from the array. Returns: Font Syntax: pop() |
| 3 | `h3_58abd0f1` | -7.00 | 0 |  | concat | Function Summary > CombinedFonts > concat The `concat` method creates a new CombinedFonts array by appending given value... |
| 4 | `h3_cf1f61ff` | -7.77 | 0 |  | Font encoding | MIF Document Statements > Character formats > Font encoding The `<FEncoding>` statement enforces font encoding priority,... |
| 5 | `h3_b480fd23` | -8.02 | 0 |  | push | Pushes the font on the array and the returns the new length of the array. Returns: number Syntax: push(value) <!-- Data ... |
| 6 | `h2_b75d640c_tab` | -8.67 | 0 |  | Font | MIF Object Reference > Font The Font table defines four indexed properties—family, variation, weight, and angle—that col... |
| 7 | `h4_ed8fc5a2` | -8.70 | 1 | [F] | Usage | Use `PgfFont` within a `Pgf` statement to override the default font for the paragraph. Use `Font` within a `FontCatalog`... |
| 8 | `h2_96cc0c95_exa` | -8.72 | 0 |  | Table selection and sizing | ```jsx var doc = app.ActiveDoc; // Get active document from FrameMaker session if(doc.ObjectValid() == true) { // Valida... |
| 9 | `h2_7a293a97` | -8.85 | 0 |  | Font | Function Summary > Font The Font function constructs a font specification using numeric indices that reference predefine... |
| 10 | `h2_185cac41_tab` | -8.88 | 0 |  | CombinedFont | MIF Object Reference > CombinedFont The CombinedFont table defines font styling parameters through numeric indices, link... |
| 11 | `h4_5801ddba_tab` | -9.50 | 0 |  | Miscellaneous information | MIF Document Statements > Character formats > PgfFont and Font statements > Miscellaneous information The `<FLocked>` bo... |
| 12 | `h3_633eaa6a_tab` | -9.53 | 0 |  | FCodes object properties | The FCodes object defines low-level keyboard and command mappings for formatting and designkit operations in Fm. Each pr... |
| 13 | `h3_c1c08403_tab` | -9.53 | 0 |  | CharFmt object properties | These properties define text styling and linguistic behavior: FontVariation and FontWeight use indexed arrays to select ... |
| 14 | `h3_73051a5c` | -9.57 | 0 |  | Font name | When a `PgfFont` or `Font` statement includes all of the family, angle, weight, and variation properties, FrameMaker ide... |
| 15 | `h3_a643ba9d` | -9.59 | 0 |  | SimpleImportFormats | Imports formats from a document to a document or a book. If you import formats to a book, the method imports formats to ... |

**Filter matches NOT in top 15** (6 chunks):

| # | Chunk ID | Heading | Content Preview |
|---|----------|---------|-----------------|
| 1 | `h2_bafc1173` | Creating and applying character formats | You can define character formats locally or store them in the Character Catalog and apply the formats to text selections... |
| 2 | `h2_fd9f0b14_exa` | Creating and applying character formats | ```mif <MIFFile 2015> # Hand generated <FontCatalog <Font <FTag `Emphasis'> <FAngle `Italic'> > # end of Font > # end of... |
| 3 | `h2_2af17751_tab` | MIF file layout | Statement,Description MIFFile,Labels the file as a MIF document file. The `MIFFile` statement is required and must be th... |
| 4 | `h2_8a97acd1` | Character formats | A character format is defined by a `PgfFont` or a `Font` statement. Character formats can be defined locally or they can... |
| 5 | `h3_68ba790d` | FontCatalog statement | MIF Document Statements > Character formats > FontCatalog statement The `FontCatalog` statement establishes the single, ... |
| 6 | `h3_49e62526` | PgfFont and Font statements | MIF Document Statements > Character formats > PgfFont and Font statements PgfFont and Font statements define character f... |

**YOUR PICK (write 1-3 chunk IDs):**
```
positive_ids: [delete this question]
```

---

## 18. q018: How can I delete all but the default colors from the color catalog?

**Filters:**
  - Filter 1: fields=['chunk_summary', 'content'], contains_any=['ColorCatalog', 'DeleteColor', 'Default colors', 'RemoveColor'], contains_all=[], max_matches=40

**Top 15 reranker-scored candidates** (filter match marked with [F]):

| Rank | Chunk ID | Score | Label | [F] | Heading | Content Preview |
|------|----------|-------|-------|-----|---------|-----------------|
| 1 | `h3_0b41e9ad` | -1.34 | 0 |  | Delete | Deletes the specified color object. See Delete under the AFrame class for more information. Returns: int Syntax: Delete(... |
| 2 | `h4_c54a55de` | -4.46 | 1 | [F] | Usage | In a MIF file, all colors are expressed as a mixture of cyan, magenta, yellow, and black. The `ColorAttribute` statement... |
| 3 | `h3_1ebd3300` | -5.97 | 1 | [F] | ColorCatalog statement | MIF Document Statements > Color > ColorCatalog statement The `ColorCatalog` statement centrally defines all colors used ... |
| 4 | `h3_26596ca7` | -6.70 | 0 |  | DeleteAllKeyDefinitions | Function Summary > KeyCatalog > DeleteAllKeyDefinitions Deletes all key definitions in a specified key catalog, clearing... |
| 5 | `h3_308d95f2` | -7.15 | 0 |  | Delete | Deletes a key catalog. See Delete under the AFrame class for more information. Returns: int Syntax: Delete() |
| 6 | `h3_50ef99b2` | -7.35 | 0 |  | NewNamedColor | Function Summary > Doc > NewNamedColor Creates and returns a named Color object using a provided string identifier. This... |
| 7 | `h3_2bab0b9a` | -7.92 | 0 |  | GetNamedColor | Function Summary > Doc > GetNamedColor Retrieves a named Color object by its string identifier. This function is a speci... |
| 8 | `h3_08868ba0` | -8.71 | 0 |  | pop | Removes the last element from the array. Returns: Tab Syntax: pop() |
| 9 | `h3_a4313269` | -8.78 | 0 |  | pop | Removes the last element from the array. Returns: Point Syntax: pop() |
| 10 | `h2_a0de7eab` | -9.00 | 1 | [F] | Color | You can assign colors to text and objects in a FrameMaker document. A FrameMaker document has a set of default colors; y... |
| 11 | `h3_a5cef710` | -9.26 | 0 |  | pop | Removes the last element from the array. Returns: string Syntax: pop() |
| 12 | `h3_d34613f8` | -9.26 | 0 |  | pop | Removes the last element from the array. Returns: int Syntax: pop() |
| 13 | `h3_256cf52a` | -9.26 | 0 |  | pop | Removes the last element from the array. Returns: int Syntax: pop() |
| 14 | `h3_526ad487` | -9.33 | 0 |  | SetProps | Function Summary > Color > SetProps Sets a color property using a PropVal object, directly modifying the color’s state. ... |
| 15 | `h3_7ffca713` | -9.33 | 0 |  | pop | Removes the last element from the array. Returns: Font Syntax: pop() |

**Filter matches NOT in top 15** (4 chunks):

| # | Chunk ID | Heading | Content Preview |
|---|----------|---------|-----------------|
| 1 | `h2_5bc2bd98` | FrameMaker documents have default objects | A FrameMaker document always has a certain set of default objects, formats, and preferences. When you create a MIF file,... |
| 2 | `h2_2af17751_tab` | MIF file layout | Statement,Description MIFFile,Labels the file as a MIF document file. The `MIFFile` statement is required and must be th... |
| 3 | `h4_be9705d9` | Usage | MIF Document Statements > Tables > TblFormat statement > Usage The `TblFormat` statement links table styling to predefin... |
| 4 | `h3_58fe98b7` | Color statement | The `Color` statement defines a color. It must appear within the `ColorCatalog` statement. Note that MIF version 5.5 and... |

**YOUR PICK (write 1-3 chunk IDs):**
```
positive_ids: [h3_1ebd3300]
```

---

## 19. q019: How can I add custom colors in RGB to the color catalog of every open file?

**Filters:**
  - Filter 1: fields=['chunk_summary', 'content'], contains_any=['AddColor', 'SetColorDefinition', 'RGB', 'Color catalog'], contains_all=[], max_matches=40

**Top 15 reranker-scored candidates** (filter match marked with [F]):

| Rank | Chunk ID | Score | Label | [F] | Heading | Content Preview |
|------|----------|-------|-------|-----|---------|-----------------|
| 1 | `h2_a0de7eab` | -2.32 | 1 | [F] | Color | You can assign colors to text and objects in a FrameMaker document. A FrameMaker document has a set of default colors; y... |
| 2 | `h3_58fe98b7` | -4.13 | 0 |  | Color statement | The `Color` statement defines a color. It must appear within the `ColorCatalog` statement. Note that MIF version 5.5 and... |
| 3 | `h2_616783de_tab` | -7.07 | 0 |  | Color | This table defines color properties in FrameMaker’s MIF system, enabling precise specification of custom and spot colors... |
| 4 | `h3_4301db27` | -9.43 | 0 |  | System generated colors | FrameMaker will automatically generate new colors when multiple tags are applied on text. The `ColorTag` tag that is gen... |
| 5 | `h3_f8bc8be6_tab` | -9.51 | 0 |  | CharFmt object properties | These properties control whether character formatting overrides defaults (UsePosition, UseStretch, etc.), each returning... |
| 6 | `h3_784f10af` | -9.70 | 0 |  | Applying a paragraph format | To apply a format from the Paragraph Catalog to a paragraph, use the `PgfTag` statement to include the format tag name w... |
| 7 | `h3_0b41e9ad` | -9.72 | 0 |  | Delete | Deletes the specified color object. See Delete under the AFrame class for more information. Returns: int Syntax: Delete(... |
| 8 | `h3_1ebd3300` | -9.73 | 1 | [F] | ColorCatalog statement | MIF Document Statements > Color > ColorCatalog statement The `ColorCatalog` statement centrally defines all colors used ... |
| 9 | `h3_526ad487` | -9.97 | 0 |  | SetProps | Function Summary > Color > SetProps Sets a color property using a PropVal object, directly modifying the color’s state. ... |
| 10 | `h3_2bab0b9a` | -9.97 | 0 |  | GetNamedColor | Function Summary > Doc > GetNamedColor Retrieves a named Color object by its string identifier. This function is a speci... |
| 11 | `h3_cb8abda0` | -10.12 | 0 |  | Applying a table format | You can apply a table format from the Table Catalog or you can define a table format locally. To apply a table format fr... |
| 12 | `h4_c54a55de` | -10.18 | 0 |  | Usage | In a MIF file, all colors are expressed as a mixture of cyan, magenta, yellow, and black. The `ColorAttribute` statement... |
| 13 | `h4_9b949854_tab` | -10.27 | 0 |  | Custom catalogs | <!-- Data Table --> MIF object,Description <CustomFontFlag (boolean)>,Yes means the document has a custom character tag ... |
| 14 | `h3_15e284ce` | -10.42 | 0 |  | GetProps | Retrieves the properties of the specified color object. See GetProps under the AFrame class for more information. Return... |
| 15 | `h3_50ef99b2` | -10.52 | 0 |  | NewNamedColor | Function Summary > Doc > NewNamedColor Creates and returns a named Color object using a provided string identifier. This... |

**Filter matches NOT in top 15** (7 chunks):

| # | Chunk ID | Heading | Content Preview |
|---|----------|---------|-----------------|
| 1 | `h3_d411eb95_tab` | Book object properties | The Book object properties control PDF generation settings for FrameMaker books, mapping UI options to programmatic cont... |
| 2 | `h2_1640b16f_tab` | Constants | These constants configure PDF export behaviors in a publishing system, defining settings like page range, dimensions, co... |
| 3 | `h3_88f1d6b8_tab` | Doc object properties | These properties control PDF generation and rendering behaviors. PDFBookmarksOpenLevel sets bookmark expansion depth; PD... |
| 4 | `h4_feb5265a_tab` | Import: syntax of strings passed to Constants.FS | This table maps format IDs to their corresponding file types for import via Constants.FS, enabling CorelDRAW to recogniz... |
| 5 | `h4_84bfcda4_tab` | PDF properties | MIF object,Description <DAcrobatBookmarksIncludeTagNames (boolean)>,Yes specifies that each PDF Bookmark title begins wi... |
| 6 | `h3_2858126b` | Generic object statements | All object descriptions consist of the object type, generic object statements containing information that is common to a... |
| 7 | `h4_40d39340_tab` | Record of the filter used to import graphic by ref... | Code,Description PICT,QuickDraw PICT WMF,Windows MetaFile EPSF,Encapsulated PostScript (Macintosh) ‘EPSI',Encapsulated P... |

**YOUR PICK (write 1-3 chunk IDs):**
```
positive_ids: [h2_a0de7eab, h3_1ebd3300, h3_58fe98b7]
```

---

## 20. q020: How can I use JSX to find all TblTitle Text and return it as a list?

**Filters:**
  - Filter 1: fields=['chunk_summary', 'content'], contains_any=['TblTitle', 'TblTitleText', 'FindTblTitle', 'table title list'], contains_all=[], max_matches=40

**Top 15 reranker-scored candidates** (filter match marked with [F]):

| Rank | Chunk ID | Score | Label | [F] | Heading | Content Preview |
|------|----------|-------|-------|-----|---------|-----------------|
| 1 | `h2_06748218_exa` | -4.19 | 0 |  | Get paragraph text | ```jsx function getText (textObj, doc) { // Gets the text from the text object or text range var textItems, text, i; // ... |
| 2 | `h3_f2d29695_tab` | -6.01 | 0 |  | Find | Function Summary > Doc > Find The Find function locates text starting from a specified text location using a property li... |
| 3 | `h3_210d8705` | -6.17 | 0 |  | QuickSelect | Implements a quick-key interface that allows the user to choose a string from a list of strings in the docu -ment Tag ar... |
| 4 | `h3_5c57bb4f` | -6.96 | 0 |  | Find | Performs the same actions as using the Find dialog box to search a document for text or other types of content. The prop... |
| 5 | `h4_c897d9cd_tab` | -7.25 | 1 | [F] | Table title | MIF Document Statements > Tables > Tbl statement > Table title The `TblTitle` statement defines a table’s title using on... |
| 6 | `h3_668ed11a_exa` | -7.35 | 0 |  | Paste | ```jsx It is illegal to specify Constants.FF_REPLACE_CELLS (0x0020)| Constants.FF_INSERT_BELOW_RIGHT (0x0008). ``` |
| 7 | `h3_9a1b2f5c` | -8.32 | 0 |  | GetTextPropVal | Gets a text property (such as the format tag, font family and size, or conditions) for a location in text. As a text pro... |
| 8 | `h3_ce91d11c` | -8.51 | 0 |  | GetText | Function Summary > Element > GetText Retrieves text content from an element using bit flags to control what text compone... |
| 9 | `h2_c911307e` | -8.53 | 0 |  | Global Methods: FDK vs JSX | Not every method is accessible through a specific object. There are some methods that are not called through any objects... |
| 10 | `h3_29f32716` | -8.69 | 0 |  | GetText | Function Summary > Flow > GetText Retrieves text content from a flow using bit flags to control what elements are includ... |
| 11 | `h3_c394d5e1` | -8.81 | 0 |  | GetText | Function Summary > SubCol > GetText Retrieves text content from a SubCol object using bit flags to control which text el... |
| 12 | `h3_1d95872b` | -8.86 | 0 |  | GetTextForRange | Gets the text for a specified text range. Call the method on the document as follows: document.GetTextForRange() You can... |
| 13 | `h2_23f03f38` | -8.89 | 0 |  | Example: Basic Object Access: FDK vs JSX | The `app` property is readily available to all FrameMaker scripts and maps to the `FO_Session` object in FDK. In this ex... |
| 14 | `h3_7093c2d5` | -8.91 | 0 |  | NewAnchoredTbl | Function Summary > Doc > NewAnchoredTbl Creates and inserts an anchored table at a specified text location, binding it t... |
| 15 | `h3_83942f5c` | -8.93 | 0 |  | Import | Imports text or graphics into a document. See Import under the Book class for more information. Returns: Object Syntax: ... |

**Filter matches NOT in top 15** (10 chunks):

| # | Chunk ID | Heading | Content Preview |
|---|----------|---------|-----------------|
| 1 | `h2_092bd847_tab` | Constants | These constants define integer flags for validating and referencing structured content in a publishing system. They cate... |
| 2 | `h2_32903cab_tab` | Constants | These constants define structured properties for tables and rulings in a document layout system. They enable programmati... |
| 3 | `h2_faa50981_tab` | Constants | These constants define table formatting behaviors in MIF, categorizing layout options like title placement (above/below/... |
| 4 | `h3_19c7aa03_tab` | Tbl object properties | The Tbl object properties define visual and structural attributes of tables, including colors, fill patterns, element as... |
| 5 | `h3_fa0e3f71_tab` | Tbl object properties | These properties define visual and structural formatting of a table, including borders, selection ranges, and alternatin... |
| 6 | `h3_2266dadc_tab` | TblFmt object properties | The TblFmt properties define table layout and positioning: alignment (left/center/right), vertical placement (page/colum... |
| 7 | `h3_fcc1c11d_tab` | TblFmt object properties | The TblFmt properties define table structure and behavior: title placement, numbering direction, initial row/column coun... |
| 8 | `h3_5f801b36_exa` | Creating a table format | ```mif <TblFormat <TblTag `Coffee Table'> # Every table must have at least one TblColumn # statement. <TblColumn <TblCol... |
| 9 | `h4_a6c4de01_tab` | Basic properties | <!-- Data Table --> MIF object,Description <TblFormat, <TblTag (tagstring)>,Table format tag name <TblLIndent (dimension... |
| 10 | `h3_c0f747cf` | Notes statement | The Notes statement defines all of the footnotes that will be used in a table title, cell, or text flow. It can appear a... |

**YOUR PICK (write 1-3 chunk IDs):**
```
positive_ids: [h3_19c7aa03_tab, h4_c897d9cd_tab]
```

---

## 21. q021: How do I extract (remove) all unresolved variables with their definitions from a FrameMaker book via MIF?

**Filters:**
  - Filter 1: fields=['chunk_summary', 'content'], contains_any=['UnresolvedVariable', 'VarFmt', 'VariableDef', 'Remove variable'], contains_all=[], max_matches=40

**Top 15 reranker-scored candidates** (filter match marked with [F]):

| Rank | Chunk ID | Score | Label | [F] | Heading | Content Preview |
|------|----------|-------|-------|-----|---------|-----------------|
| 1 | `h2_2d0830da` | 0.78 | 0 |  | Creating variables | In a FrameMaker document, variables act as placeholders for text that might change. For example, many documents use a va... |
| 2 | `h3_7f860344_exa` | -1.83 | 0 |  | Editing the MIF file | ```mif <Page <Unique 45155> <PageType BodyPage > <PageNum `1'> <PageSize 8.5" 11.0"> <PageOrientation Portrait > <PageAn... |
| 3 | `h3_56520ef7` | -3.45 | 1 | [F] | Using system variables | Whenever you open or import a MIF file, the MIF interpreter provides the default system variables. You can redefine a sy... |
| 4 | `h2_d1563e82` | -3.89 | 0 |  | Creating cross-references | In a FrameMaker document, you can create cross-references that are automatically updated. A cross-reference can refer to... |
| 5 | `h3_d8665f70` | -4.07 | 0 |  | Adding headers and footers | Headers and footers are defined in untagged text flows on the master pages of a document. When FrameMaker creates defaul... |
| 6 | `h2_4883b66c` | -4.49 | 0 |  | Variables | MIF Document Statements > Variables Variable definitions in a MIF document are centralized under `VariableFormats`, enco... |
| 7 | `h2_ad11e077` | -4.84 | 0 |  | How FrameMaker identifies MIF files | MIF overview > How FrameMaker identifies MIF files FrameMaker identifies MIF files by the presence of a `MIFFile` or `Bo... |
| 8 | `h3_f30c00eb` | -4.95 | 0 |  | Tips | The following hints may help you minimize the MIF statements for paragraph formats: * If possible, use the formats in th... |
| 9 | `h3_c379d146` | -5.05 | 0 |  | How FrameMaker writes cross-references | When FrameMaker writes a cross-reference, it provides the actual text that will appear at the reference point. This info... |
| 10 | `h3_56b8bdd9` | -5.81 | 1 | [F] | Usage | MIF Document Statements > Variables > Usage Variables in MIF are named via `VariableName` and referenced by `Variable` t... |
| 11 | `h2_5fba4ca6` | -5.90 | 0 |  | Including template files | When you write an application, such as a filter or a database publishing application, to generate a MIF file, you have t... |
| 12 | `h2_7e332fd1` | -5.92 | 0 |  | Creating filters | Structured FrameMaker allows specific components in a structured document to be processed differently to generate differ... |
| 13 | `h3_8fdf0de5` | -6.05 | 1 | [F] | Defining user variables | All variable definitions for a document are contained in a single `VariableFormats` statement. The `VariableFormats` sta... |
| 14 | `h2_d03606f7` | -6.16 | 0 |  | MIF file layout | MIF Document Statements > MIF file layout FrameMaker writes MIF files in a strict structural order, ensuring consistency... |
| 15 | `h2_b4e454fd` | -6.29 | 0 |  | Graphic objects and graphic frames | In a FrameMaker document, graphic objects can appear directly on a page or within a graphic frame. The following objects... |

**Filter matches NOT in top 15** (13 chunks):

| # | Chunk ID | Heading | Content Preview |
|---|----------|---------|-----------------|
| 1 | `h2_11a19fea_tab` | Constants | These constants define internal identifiers for frame variables and formatting properties used in document processing, p... |
| 2 | `h2_b0f692dc_tab` | Constants | These constants define unique integer identifiers for frame, footnote, marker, and variable properties in FramerScript. ... |
| 3 | `h2_3629a7b6_tab` | Constants | These constants define numeric identifiers for key document elements and states in FrameMaker’s API, enabling programmat... |
| 4 | `h2_01291df9_tab` | Constants | These constants define object types in the FrameMaker API, each mapped to a unique integer ID for programmatic identific... |
| 5 | `h3_0f3c0b00_tab` | Doc object properties | These properties expose key document elements for programmatic access, enabling navigation and manipulation of core comp... |
| 6 | `h3_3e1957a3` | Doc methods | AddNewBuildExpr, AddText, CenterOnText, Clear, ClearAllChangebars, Close, Compare, Copy, Cut, DeleteBuildExpr, DeleteTex... |
| 7 | `h2_42620f51_tab` | Var | This table defines properties of a Var object, linking it to FrameMaker’s document structure. Each variable carries its ... |
| 8 | `h2_a97f3127_tab` | VarFmt | The VarFmt table defines template structures for dynamic variables in documents. Each entry specifies a format string (F... |
| 9 | `h3_ecc499f0` | GetNamedObject | Gets the object with the specified name and type. The method works with the following objects: - AttrCondExpr - CharFmt ... |
| 10 | `h3_845ffe57` | GetNamedVarFmt | Function Summary > Doc > GetNamedVarFmt Retrieves a named Variable Format object by its identifier, enabling access to f... |
| ... | (3 more) | | |

**YOUR PICK (write 1-3 chunk IDs):**
```
positive_ids: [delete this question]
```

---

## 22. q024: Can I globally search and replace table title text in all open files?

**Filters:**
  - Filter 1: fields=['chunk_summary', 'content'], contains_any=['TblTitle', 'FindReplace', 'table title text', 'ReplaceTblTitle'], contains_all=[], max_matches=40

**Top 15 reranker-scored candidates** (filter match marked with [F]):

| Rank | Chunk ID | Score | Label | [F] | Heading | Content Preview |
|------|----------|-------|-------|-----|---------|-----------------|
| 1 | `h4_c897d9cd_tab` | -6.17 | 1 | [F] | Table title | MIF Document Statements > Tables > Tbl statement > Table title The `TblTitle` statement defines a table’s title using on... |
| 2 | `h3_5c57bb4f` | -8.16 | 0 |  | Find | Performs the same actions as using the Find dialog box to search a document for text or other types of content. The prop... |
| 3 | `h4_4bc567e5_tab` | -8.98 | 0 |  | Basic properties | <!-- Data Table --> |
| 4 | `h4_d8b20654_tab` | -8.98 | 0 |  | PDF properties | <!-- Data Table --> |
| 5 | `h3_cfca9f38` | -9.24 | 0 |  | Locked tables and text insets | The `TblLocked` statement does not correspond to any setting in the Table Designer. The statement is for text insets tha... |
| 6 | `h2_48f243f2` | -9.84 | 0 |  | TblFmt | The method uses an FO\_TblFmt object to represent each table format in a document. |
| 7 | `h2_7f5a1ad7_exa` | -10.06 | 0 |  | Verify hypertext marker links | ```jsx // Process all open documents in a book function checkHypertextLinks_Book(book) { // Initialize book-level report... |
| 8 | `h3_f2d29695_tab` | -10.11 | 0 |  | Find | Function Summary > Doc > Find The Find function locates text starting from a specified text location using a property li... |
| 9 | `h4_c4bcbc6a_tab` | -10.33 | 0 |  | Miscellaneous properties | <!-- Data Table --> MIF object,Description <TblLocked (boolean)>,Yes means the table is part of a text inset that obtain... |
| 10 | `h2_ef785cf7_exa` | -10.36 | 0 |  | Verify hypertext marker links | ```jsx // Find open document or book by file path with flexible matching options function getOpenFile(path, considerSubs... |
| 11 | `h3_fb1188a6` | -10.43 | 0 |  | Tbls statement | MIF Document Statements > Tables > Tbls statement The `Tbls` statement declares all tables in a MIF document, serving as... |
| 12 | `h3_4aa97070` | -10.50 | 0 |  | Tbl statement | The `Tbl` statement contains the contents of a table instance. It must appear in a `Tbls` statement. Each `Tbl` statemen... |
| 13 | `h2_2fddfa5d_exa` | -10.51 | 0 |  | Get all open documents | ```jsx // Get first document from FrameMaker's internal document stack (unordered) var doc = app.FirstOpenDoc; var openD... |
| 14 | `h3_627997b4` | -10.58 | 0 |  | Creating a table instance | All table instances in a document are contained in a `Tbls` statement. The `Tbls` statement contains a list of `Tbl` sta... |
| 15 | `h4_81ba3866_tab` | -10.60 | 0 |  | Table footnote properies | MIF Document Statements > Global document properties > Document statement > Table footnote properies These table footnot... |

**Filter matches NOT in top 15** (10 chunks):

| # | Chunk ID | Heading | Content Preview |
|---|----------|---------|-----------------|
| 1 | `h2_092bd847_tab` | Constants | These constants define integer flags for validating and referencing structured content in a publishing system. They cate... |
| 2 | `h2_32903cab_tab` | Constants | These constants define structured properties for tables and rulings in a document layout system. They enable programmati... |
| 3 | `h2_faa50981_tab` | Constants | These constants define table formatting behaviors in MIF, categorizing layout options like title placement (above/below/... |
| 4 | `h3_19c7aa03_tab` | Tbl object properties | The Tbl object properties define visual and structural attributes of tables, including colors, fill patterns, element as... |
| 5 | `h3_fa0e3f71_tab` | Tbl object properties | These properties define visual and structural formatting of a table, including borders, selection ranges, and alternatin... |
| 6 | `h3_2266dadc_tab` | TblFmt object properties | The TblFmt properties define table layout and positioning: alignment (left/center/right), vertical placement (page/colum... |
| 7 | `h3_fcc1c11d_tab` | TblFmt object properties | The TblFmt properties define table structure and behavior: title placement, numbering direction, initial row/column coun... |
| 8 | `h3_5f801b36_exa` | Creating a table format | ```mif <TblFormat <TblTag `Coffee Table'> # Every table must have at least one TblColumn # statement. <TblColumn <TblCol... |
| 9 | `h4_a6c4de01_tab` | Basic properties | <!-- Data Table --> MIF object,Description <TblFormat, <TblTag (tagstring)>,Table format tag name <TblLIndent (dimension... |
| 10 | `h3_c0f747cf` | Notes statement | The Notes statement defines all of the footnotes that will be used in a table title, cell, or text flow. It can appear a... |

**YOUR PICK (write 1-3 chunk IDs):**
```
positive_ids: [delete this question]
```

---

## 23. q025: How can I script adding file metadata to a book so that I don't have to manually do it from the File Info menu?

**Filters:**
  - Filter 1: fields=['chunk_summary', 'content'], contains_any=['FileInfo', 'DocSummary', 'SetUserString', 'Document metadata'], contains_all=[], max_matches=40

**Top 15 reranker-scored candidates** (filter match marked with [F]):

| Rank | Chunk ID | Score | Label | [F] | Heading | Content Preview |
|------|----------|-------|-------|-----|---------|-----------------|
| 1 | `h2_d42e7f60_tab` | -8.08 | 0 |  | Constants | These constants define book-related file types and properties in a publishing system, using bit flags (e.g., FV_BK_BOOK ... |
| 2 | `h4_bd2f3d33` | -8.17 | 1 | [F] | Document File Info | MIF Document Statements > Global document properties > Document statement > Document File Info FrameMaker 7.0+ embeds do... |
| 3 | `h3_19a5753b_exa` | -8.62 | 0 |  | Configuration and Global Variables | Report document properties > Create a favorites list > Configuration and Global Variables Configures where Favorites app... |
| 4 | `h4_5d7b9ef6` | -8.94 | 0 |  | PDF Document Info | For versions 6.0 and later, FrameMaker stores PDF File Info in the document file. FrameMaker automatically supplies valu... |
| 5 | `h3_dcda5882` | -9.62 | 0 |  | SimpleSave | The SimpleSave() method saves a book. If you set the interactive parameter to False and you specify the book's current n... |
| 6 | `h3_ea62e5d2` | -9.72 | 0 |  | ManageConditionalExpressions | Add, edit, or delete conditional expression tags to the current book. Applies to the options available in the Add/Edit C... |
| 7 | `h3_d8665f70` | -9.79 | 0 |  | Adding headers and footers | Headers and footers are defined in untagged text flows on the master pages of a document. When FrameMaker creates defaul... |
| 8 | `h3_b64696d3` | -9.82 | 0 |  | SimpleSave | Saves a document or book. If you set the interactive parameter to False and specify the document or book's current name ... |
| 9 | `h3_d44c43cd` | -9.91 | 0 |  | UpdateBook | The UpdateBook() method updates a book. The method allows you to specify a script (property list) specifying how to upda... |
| 10 | `h3_a643ba9d` | -10.00 | 0 |  | SimpleImportFormats | Imports formats from a document to a document or a book. If you import formats to a book, the method imports formats to ... |
| 11 | `h3_f30b8b09` | -10.07 | 0 |  | NewSeriesBookComponent | Function Summary > Book > NewSeriesBookComponent NewSeriesBookComponent() inserts a new Book Component into a series at ... |
| 12 | `h3_e3ea2f94` | -10.09 | 0 |  | Editing the MIF file | Using MIF Statements > Including template files > Editing the MIF file Edit the MIF file to isolate formatting and layou... |
| 13 | `h3_ecbce3f2` | -10.11 | 0 |  | Save | The Save() method saves a book. The method allows you to script the way FrameMaker saves the file and to specify respons... |
| 14 | `h3_ae5a1fd6_tab` | -10.11 | 0 |  | Book object properties | These properties manage XML metadata and encoding for Book objects. XmlStyleSheet and XmlStyleSheetList define CSS style... |
| 15 | `h3_210d8705` | -10.13 | 1 | [F] | QuickSelect | Implements a quick-key interface that allows the user to choose a string from a list of strings in the docu -ment Tag ar... |

**Filter matches NOT in top 15** (8 chunks):

| # | Chunk ID | Heading | Content Preview |
|---|----------|---------|-----------------|
| 1 | `h1_019ab415_exa` | Report document properties | ```jsx var doc = app.ActiveDoc; if(doc.ObjectValid() == true) { // Display introductory information with document filena... |
| 2 | `h3_d411eb95_tab` | Book object properties | The Book object properties control PDF generation settings for FrameMaker books, mapping UI options to programmatic cont... |
| 3 | `h3_d2956be0_tab` | Book object properties | The Book object properties define runtime window behavior and display settings. StatusLine, ScreenX/Y, and ScreenHeight/... |
| 4 | `h2_1eecbb9a_tab` | Constants | These constants define operational modes and file type identifiers for system UI and document handling. Grouped into fil... |
| 5 | `h2_e2c2b5e8_tab` | Constants | These constants define system and document metadata values for dynamic field substitution in MIF documents. They enable ... |
| 6 | `h2_b422493d_tab` | Constants | These constants define PDF-specific configuration options for document properties, bookmarks, job settings, and zoom beh... |
| 7 | `h3_9a6d8b27_tab` | Doc object properties | These properties control document metadata and formatting behaviors, particularly for multi-volume documents and change ... |
| 8 | `h4_84bfcda4_tab` | PDF properties | MIF object,Description <DAcrobatBookmarksIncludeTagNames (boolean)>,Yes specifies that each PDF Bookmark title begins wi... |

**YOUR PICK (write 1-3 chunk IDs):**
```
positive_ids: [h4_bd2f3d33, h4_5d7b9ef6, h4_a668e400_exa]
```

---

## 24. q026: What script iterates all text insets and reports missing external source files?

**Filters:**
  - Filter 1: fields=['chunk_summary', 'content'], contains_any=['TiApiClient', 'TiFile', 'TiAutomaticUpdate', 'NextTiInDoc'], contains_all=[], max_matches=40
  - Filter 2: fields=['chunk_summary', 'content'], contains_any=['TiText', 'text inset', 'TiClientData'], contains_all=[], max_matches=40

**Top 15 reranker-scored candidates** (filter match marked with [F]):

| Rank | Chunk ID | Score | Label | [F] | Heading | Content Preview |
|------|----------|-------|-------|-----|---------|-----------------|
| 1 | `h2_6ca8f250_exa` | -0.74 | 0 |  | Write paragraph text to the console | JSX example scripts > Write paragraph text to the console This script extracts and logs all paragraph text from the acti... |
| 2 | `h2_2fddfa5d_exa` | -0.99 | 0 |  | Get all open documents | ```jsx // Get first document from FrameMaker's internal document stack (unordered) var doc = app.FirstOpenDoc; var openD... |
| 3 | `h2_ead2c852_exa` | -2.00 | 0 |  | Retrieve text over a network | ```jsx var doc = app.ActiveDoc; // Active document if (doc.ObjectValid()) { alert("Contacting www.test.com/resources/es_... |
| 4 | `h2_7880444b` | -2.47 | 0 |  | Get all open documents | This script demonstrates session-level document navigation by iterating through all open documents in FrameMaker and rep... |
| 5 | `h2_cb50fcf2_exa` | -2.70 | 0 |  | Report all paragraph formats | ```jsx var doc = app.ActiveDoc; // Get active document reference if(doc.ObjectValid() == true) { // Validate active docu... |
| 6 | `h2_26703f2f_tab` | -3.91 | 1 | [F] | TiText | TiText objects manage imported text insets in FrameMaker, controlling how external files are embedded and updated. Key p... |
| 7 | `h2_06748218_exa` | -4.25 | 0 |  | Get paragraph text | ```jsx function getText (textObj, doc) { // Gets the text from the text object or text range var textItems, text, i; // ... |
| 8 | `h3_6ddc74f3_exa` | -4.53 | 0 |  | Error Handling and Completion | ```jsx // Handle missing document or condition format else { alert("No active document found or the active document does... |
| 9 | `h2_753b4e07` | -4.99 | 0 |  | Retrieve text over a network | This script demonstrates network communication by retrieving text content from a remote web server and replacing paragra... |
| 10 | `h2_115faa40_exa` | -5.85 | 0 |  | Report paragraphs and their formats | ```jsx var pgfText = ""; // Initialize text content variable var msg = ""; // Initialize progressive message string var ... |
| 11 | `h2_4a06cb76_exa` | -5.92 | 0 |  | Change paragraph format font size | ```jsx var formatName = "Body"; // Set target format name for modification var newFontSize = 24; // Set desired font siz... |
| 12 | `h2_43d20e69` | -6.79 | 0 |  | Changing Table Properties | The following script changes the left indentation of all the tables in a FrameMaker body page by 1 inch and the width of... |
| 13 | `h3_a219b05a_tab` | -7.88 | 1 | [F] | Doc object properties | These properties control document behavior during save and open operations. DocSaveType defines the output format (binar... |
| 14 | `h2_925caf96` | -7.94 | 0 |  | Applying conditions based on paragraph formats | This script demonstrates conditional text application and paragraph processing by systematically iterating through docum... |
| 15 | `h2_0bf04d3e` | -8.10 | 0 |  | Book operations: navigation, opening, updating | This script demonstrates comprehensive book management by navigating book components, opening files with robust error ha... |

**Filter matches NOT in top 15** (38 chunks):

| # | Chunk ID | Heading | Content Preview |
|---|----------|---------|-----------------|
| 1 | `h2_8ff78ef7_tab` | Constants | These constants define configuration flags for managing TI (Text Instruction) objects in a document system. Integer valu... |
| 2 | `h2_27593350_tab` | Constants | These constants define core operational modes and flags for document handling, command states, and text item behavior in... |
| 3 | `h2_3c9577d0_tab` | Constants | These constants define integer identifiers for formatting and object types within the MIF system, enabling precise type ... |
| 4 | `h3_0f3c0b00_tab` | Doc object properties | These properties expose key document elements for programmatic access, enabling navigation and manipulation of core comp... |
| 5 | `h3_dd58b742_tab` | Doc object properties | These properties expose the first instance of key document components—flows, paragraphs, graphics, markers, formats, and... |
| 6 | `h3_3e1957a3` | Doc methods | AddNewBuildExpr, AddText, CenterOnText, Clear, ClearAllChangebars, Close, Compare, Copy, Cut, DeleteBuildExpr, DeleteTex... |
| 7 | `h3_9714de3e_tab` | TextItem types | TextItem types define structural and formatting anchors within text flows, mapping logical document elements (paragraphs... |
| 8 | `h2_3e2fc61c_tab` | TiApiClient | The TiApiClient properties define metadata and behavior for text insets in FrameMaker, enabling clients to manage sourci... |
| 9 | `h2_b86b62d0_tab` | TiFlow | TiFlow defines properties for imported text flows in FrameMaker, controlling how source content is embedded, formatted, ... |
| 10 | `h2_9e331b99_tab` | TiTextTable | TiTextTable manages imported tabular text, converting paragraphs into rows or cells based on TiByRows. It defines struct... |
| ... | (28 more) | | |

**YOUR PICK (write 1-3 chunk IDs):**
```
positive_ids: [h2_feb1f085_tab, h3_0f3c0b00_tab]
```

---

## 25. q029: How can I auto-create cross-references for glossary entries detected in body text?

**Filters:**
  - Filter 1: fields=['chunk_summary', 'content'], contains_any=['XRefSrcText', 'XRefFmt', 'XRefClientName', 'XRefIsUnresolved'], contains_all=[], max_matches=60
  - Filter 2: fields=['chunk_summary', 'content'], contains_any=['FirstXRefInDoc', 'XRefFile', 'XRefAltText'], contains_all=[], max_matches=40

**Top 15 reranker-scored candidates** (filter match marked with [F]):

| Rank | Chunk ID | Score | Label | [F] | Heading | Content Preview |
|------|----------|-------|-------|-----|---------|-----------------|
| 1 | `h2_d1563e82` | 2.02 | 0 |  | Creating cross-references | In a FrameMaker document, you can create cross-references that are automatically updated. A cross-reference can refer to... |
| 2 | `h3_c379d146` | 0.65 | 1 | [F] | How FrameMaker writes cross-references | When FrameMaker writes a cross-reference, it provides the actual text that will appear at the reference point. This info... |
| 3 | `h3_b24ec725` | 0.44 | 1 | [F] | Inserting the reference point | The final step in creating a cross-reference is to insert an `XRef` statement at the position in text where the cross-re... |
| 4 | `h3_f7142515` | -2.51 | 0 |  | Inserting the reference source marker | To mark the location of the reference source, insert a `Marker` statement at the beginning of the reference source. The ... |
| 5 | `h2_6ad963d1` | -3.29 | 0 |  | Creating markers | A FrameMaker document can contain markers that hold hidden text and mark locations. For example, you use markers to add ... |
| 6 | `h3_845648d6` | -3.71 | 1 | [F] | NewNamedXRefFmt | Function Summary > Doc > NewNamedXRefFmt Creates a named Cross Reference Format for consistent document referencing. Ass... |
| 7 | `h3_321c2e0e` | -3.85 | 0 |  | NewAnchoredFormattedXRef | Function Summary > Doc > NewAnchoredFormattedXRef Creates an anchored, formatted cross-reference tied to a specific text... |
| 8 | `h2_240a890d` | -3.93 | 0 |  | Cross-references | MIF Document Statements > Cross-references FrameMaker documents use cross-references to link internal or external conten... |
| 9 | `h3_990893f8` | -4.68 | 1 | [F] | GetNamedXRefFmt | Function Summary > Doc > GetNamedXRefFmt Retrieves a named Cross Reference Format object by its identifier, enabling con... |
| 10 | `h3_ffa3d1d6` | -5.12 | 0 |  | UpdateXRef | Updates the cross-references in a document. It performs the same operation as clicking Update in the Cross-Reference win... |
| 11 | `h3_75f8881c` | -5.12 | 0 |  | Creating cross-reference formats | The cross-reference formats for a document are defined in one `XRefFormats` statement. A document can have only one `XRe... |
| 12 | `h2_be6149de_tab` | -5.51 | 1 | [F] | XRef | The XRef table defines properties for cross-references in FrameMaker, linking elements to their source and resolution st... |
| 13 | `h3_e98fdb75` | -5.83 | 0 |  | XRefFormats and XRefFormat statements | The `XRefFormats` statement defines the formats of cross-references to be used in document text flows. A MIF file can ha... |
| 14 | `h4_036d3a4b_tab` | -5.96 | 0 |  | Reference properties | MIF Document Statements > Global document properties > Document statement > Reference properties These properties enable... |
| 15 | `h2_88705d5b_tab` | -6.88 | 1 | [F] | XRef | The XRef table defines cross-reference metadata linking source content to client-generated references. It distinguishes ... |

**Filter matches NOT in top 15** (9 chunks):

| # | Chunk ID | Heading | Content Preview |
|---|----------|---------|-----------------|
| 1 | `h2_a42181a1_tab` | Constants | These constants define text formatting and document behavior options in a publishing system. They control line alignment... |
| 2 | `h2_11a19fea_tab` | Constants | These constants define internal identifiers for frame variables and formatting properties used in document processing, p... |
| 3 | `h2_3629a7b6_tab` | Constants | These constants define numeric identifiers for key document elements and states in FrameMaker’s API, enabling programmat... |
| 4 | `h2_01291df9_tab` | Constants | These constants define object types in the FrameMaker API, each mapped to a unique integer ID for programmatic identific... |
| 5 | `h3_0f3c0b00_tab` | Doc object properties | These properties expose key document elements for programmatic access, enabling navigation and manipulation of core comp... |
| 6 | `h3_3e1957a3` | Doc methods | AddNewBuildExpr, AddText, CenterOnText, Clear, ClearAllChangebars, Close, Compare, Copy, Cut, DeleteBuildExpr, DeleteTex... |
| 7 | `h2_107038ed_tab` | XRefFmt | MIF Object Reference > XRefFmt The XRefFmt table defines reusable cross-reference formats within a document. Each entry ... |
| 8 | `h3_ecc499f0` | GetNamedObject | Gets the object with the specified name and type. The method works with the following objects: - AttrCondExpr - CharFmt ... |
| 9 | `h3_de222528` | NewNamedObject | Creates the following named objects: - AttrCondExpr - CharFmt - CombinedFontDefn - Color - Command - CondFmt - ElementDe... |

**YOUR PICK (write 1-3 chunk IDs):**
```
positive_ids: [h2_88705d5b_tab, h3_754c2a4c]
```

---

## 26. q033: What MIF tokens control table ruling styles, and how can I toggle them via script?

**Filters:**
  - Filter 1: fields=['chunk_summary', 'content'], contains_any=['RulingFmt', 'CellDefaultTopRuling', 'CellOverrideBottomRuling', 'CellUseOverrideRuling'], contains_all=[], max_matches=50
  - Filter 2: fields=['chunk_summary', 'content'], contains_any=['Constants.FP_FirstRulingFmtInDoc', 'RulingPenWidth', 'TblFmt'], contains_all=[], max_matches=50

**Top 15 reranker-scored candidates** (filter match marked with [F]):

| Rank | Chunk ID | Score | Label | [F] | Heading | Content Preview |
|------|----------|-------|-------|-----|---------|-----------------|
| 1 | `h4_09495175_tab` | 0.16 | 0 |  | Ruling properties | <!-- Data Table --> MIF object,Description <TblColumnRuling (tagstring)>,Ruling style for most columns; value must match... |
| 2 | `h3_47904da0` | -0.52 | 0 |  | RulingCatalog statement | MIF Document Statements > Tables > RulingCatalog statement The `RulingCatalog` statement uniquely defines all ruling sty... |
| 3 | `h3_a8bad948` | -2.96 | 0 |  | Adding a Table Catalog | You can store table formats in a Table Catalog by using a `TblCatalog` statement. A document can have only one `TblCatal... |
| 4 | `h2_b276d889` | -4.76 | 0 |  | Tables | MIF Document Statements > Tables Table formats, rulings, and instances in MIF are centrally managed: `TblFormat` defines... |
| 5 | `h3_280e2756` | -4.85 | 1 | [F] | Ruling statement | The `Ruling` statement defines the ruling styles used in table formats. It must appear within the `RulingCatalog` statem... |
| 6 | `h3_fa0e3f71_tab` | -5.34 | 1 | [F] | Tbl object properties | These properties define visual and structural formatting of a table, including borders, selection ranges, and alternatin... |
| 7 | `h3_5f801b36_exa` | -5.69 | 0 |  | Creating a table format | ```mif <TblFormat <TblTag `Coffee Table'> # Every table must have at least one TblColumn # statement. <TblColumn <TblCol... |
| 8 | `h3_46143912_exa` | -5.76 | 0 |  | Configuration and Initialization | Report document properties > Configure attribute displays > Configuration and Initialization The script defines keyboard... |
| 9 | `h2_f4f12ee6` | -6.34 | 0 |  | MIF statement syntax | The statement descriptions in this manual use the following conventions to describe syntax: `<token data>` `token data` ... |
| 10 | `h2_46910b43` | -6.39 | 0 |  | Line numbers | FrameMaker documents can have the line numbers displayed for assisting in the reviewing process. Multiple contributors t... |
| 11 | `h2_faa50981_tab` | -6.61 | 0 |  | Constants | These constants define table formatting behaviors in MIF, categorizing layout options like title placement (above/below/... |
| 12 | `h3_09b7862d_tab` | -6.70 | 1 | [F] | TblFmt object properties | These properties define table border and ruling patterns, enabling precise control over visual structure. They specify w... |
| 13 | `h4_d5c3d2c9_tab` | -7.06 | 0 |  | Pagination | <!-- Data Table --> MIF object,Description <DStartPage (integer)>,Starting page number <DPageNumStyle (keyword)>,"Page n... |
| 14 | `h4_f2f4ad45` | -7.16 | 0 |  | Rotated cells | Using MIF Statements > Creating and applying character formats > Adding a table anchor > Rotated cells Rotated cells in ... |
| 15 | `h4_782e215c_tab` | -7.78 | 0 |  | View only document properties | <!-- Data Table --> MIF object,Description <DViewOnly (boolean)>,Yes specifies View Only document (locked) <DViewOnlyXRe... |

**Filter matches NOT in top 15** (27 chunks):

| # | Chunk ID | Heading | Content Preview |
|---|----------|---------|-----------------|
| 1 | `h3_cc98b168_tab` | Cell object properties | These properties define a cell’s spatial, structural, and visual behavior within a table. They enable dynamic access to ... |
| 2 | `h3_d5e3d476_tab` | Cell object properties | These properties enable granular control over cell appearance by overriding table-level formatting. Each override proper... |
| 3 | `h2_32903cab_tab` | Constants | These constants define structured properties for tables and rulings in a document layout system. They enable programmati... |
| 4 | `h2_f92e2c86_tab` | Constants | These constants define cell and row formatting behaviors in a table layout system, enabling precise control over borders... |
| 5 | `h2_f0e1204d_tab` | Constants | These constants define cell-level properties and navigation behaviors in a table structure, enabling precise control ove... |
| 6 | `h2_a9f5009c_tab` | Constants | These constants define document-specific identifiers and flags for FrameMaker’s internal object model, enabling programm... |
| 7 | `h2_01291df9_tab` | Constants | These constants define object types in the FrameMaker API, each mapped to a unique integer ID for programmatic identific... |
| 8 | `h3_dd58b742_tab` | Doc object properties | These properties expose the first instance of key document components—flows, paragraphs, graphics, markers, formats, and... |
| 9 | `h3_3e1957a3` | Doc methods | AddNewBuildExpr, AddText, CenterOnText, Clear, ClearAllChangebars, Close, Compare, Copy, Cut, DeleteBuildExpr, DeleteTex... |
| 10 | `h2_3578d524_tab` | RulingFmt | RulingFmt defines formatting for document ruling lines, controlling appearance via pen patterns, line count, thickness, ... |
| ... | (17 more) | | |

**YOUR PICK (write 1-3 chunk IDs):**
```
positive_ids: [h4_09495175_tab]
```

---

## 27. q036: How can I change the header color of all table header rows?

**Filters:**
  - Filter 1: fields=['chunk_summary', 'content'], contains_any=['CellOverrideShading', 'CellOverrideFill', 'CellUseOverrideShading'], contains_all=[], max_matches=40
  - Filter 2: fields=['chunk_summary', 'content'], contains_any=['TblRow', 'TblFmt', 'Header row'], contains_all=[], max_matches=40

**Top 15 reranker-scored candidates** (filter match marked with [F]):

| Rank | Chunk ID | Score | Label | [F] | Heading | Content Preview |
|------|----------|-------|-------|-----|---------|-----------------|
| 1 | `h3_db7ecd02` | -4.32 | 1 | [F] | DeleteRows | Deletes rows from a table. Like the Delete command in the FrameMaker product user interface, Delete -Rows() does not all... |
| 2 | `h2_43d20e69` | -6.73 | 0 |  | Changing Table Properties | The following script changes the left indentation of all the tables in a FrameMaker body page by 1 inch and the width of... |
| 3 | `h3_d1bb5751` | -7.06 | 1 | [F] | NewTable | Inserts a table ( FO\_Tbl object ). When you create a table in the user interface, you can specify a Table Catalog forma... |
| 4 | `h3_ecc45b55_exa` | -7.08 | 1 | [F] | Creating a table instance | ```mif <Cell # First cell in row <CellContent <Para # Cells can contain paragraphs <PgfTag `CellHeading'> # Applies form... |
| 5 | `h3_9b751453_exa` | -7.48 | 0 |  | Creating a table instance | ```mif <Tbl <TblID…> # A unique ID for the table <TblFormat…> # The table format <TblNumColumns…> # Number of columns in... |
| 6 | `h2_faa50981_tab` | -8.41 | 0 |  | Constants | These constants define table formatting behaviors in MIF, categorizing layout options like title placement (above/below/... |
| 7 | `h3_3a214cb4` | -8.53 | 0 |  | AddRows | Adds one or more rows to a table. The following table lists the constants you can specify for the direction parameter: <... |
| 8 | `h3_82339bf5` | -8.54 | 0 |  | Row statement | A Row statement contains a list of cells. It also includes row properties as needed. The statement must appear in a `Tbl... |
| 9 | `h3_535c7717_tab` | -8.68 | 1 | [F] | TblFmt object properties | The TblFmt properties define visual styling and layout rules for tables, including borders (TblRightRuling, TblTopRuling... |
| 10 | `h4_f2f4ad45` | -8.68 | 0 |  | Rotated cells | Using MIF Statements > Creating and applying character formats > Adding a table anchor > Rotated cells Rotated cells in ... |
| 11 | `h4_e3793e90` | -8.81 | 0 |  | Table rows | MIF Document Statements > Tables > Tbl statement > Table rows Table rows are organized into three logical sections: head... |
| 12 | `h2_645218e5_tab` | -8.91 | 1 | [F] | Constants | These constants define text encoding standards and document formatting behaviors for import/export operations. Encoding ... |
| 13 | `h3_64fe009c_tab` | -9.01 | 1 | [F] | NewTable | NewTable creates a table with customizable structure using a format template. It requires explicit specification of colu... |
| 14 | `h3_09b7862d_tab` | -9.22 | 1 | [F] | TblFmt object properties | These properties define table border and ruling patterns, enabling precise control over visual structure. They specify w... |
| 15 | `h2_bec2a153_tab` | -9.31 | 0 |  | Row | This row object defines structural and formatting properties for table rows in FrameMaker, controlling visibility, layou... |

**Filter matches NOT in top 15** (22 chunks):

| # | Chunk ID | Heading | Content Preview |
|---|----------|---------|-----------------|
| 1 | `h3_21710753_tab` | Cell object properties | These properties define a cell’s formatting overrides, structural relationships, and metadata within a table. They contr... |
| 2 | `h3_d5e3d476_tab` | Cell object properties | These properties enable granular control over cell appearance by overriding table-level formatting. Each override proper... |
| 3 | `h2_f0e1204d_tab` | Constants | These constants define cell-level properties and navigation behaviors in a table structure, enabling precise control ove... |
| 4 | `h2_33e79dd3_tab` | Constants | These constants define precise typographic and layout behaviors for text flows and table cells in MIF. They control sync... |
| 5 | `h2_27595b92` | Table selection and sizing | This script demonstrates comprehensive table manipulation techniques in FrameMaker, including table identification, cell... |
| 6 | `h2_95777902_tab` | Constants | These constants define layout behaviors for tables and rows in document formatting. Row positioning (e.g., top of page, ... |
| 7 | `h2_a9f5009c_tab` | Constants | These constants define document-specific identifiers and flags for FrameMaker’s internal object model, enabling programm... |
| 8 | `h2_01291df9_tab` | Constants | These constants define object types in the FrameMaker API, each mapped to a unique integer ID for programmatic identific... |
| 9 | `h2_632c8fcb_tab` | Constants | These constants define configuration flags for saving operations in MIF, controlling auto-backup behavior, page count ru... |
| 10 | `h3_0f3c0b00_tab` | Doc object properties | These properties expose key document elements for programmatic access, enabling navigation and manipulation of core comp... |
| ... | (12 more) | | |

**YOUR PICK (write 1-3 chunk IDs):**
```
positive_ids: [h4_f6dc5631_tab]
```

---

## 28. q039: How do I enable automatic paragraph numbering with customizable formatting?

**Filters:**
  - Filter 1: fields=['chunk_summary', 'content'], contains_any=['PgfIsAutoNum', 'PgfNumber', 'NumAtEnd'], contains_all=[], max_matches=40
  - Filter 2: fields=['chunk_summary', 'content'], contains_any=['FmtChangeList', 'PgfFmt', 'PgfCatalog'], contains_all=[], max_matches=40

**Top 15 reranker-scored candidates** (filter match marked with [F]):

| Rank | Chunk ID | Score | Label | [F] | Heading | Content Preview |
|------|----------|-------|-------|-----|---------|-----------------|
| 1 | `h3_f44bc933` | -7.10 | 1 | [F] | GetNamedFmtChangeList | Function Summary > Doc > GetNamedFmtChangeList Retrieves a named Format Change List object from the document. This funct... |
| 2 | `h2_c1f016cc_tab` | -7.67 | 0 |  | Constants | This table defines page numbering styles and document properties as constants, primarily for controlling how page number... |
| 3 | `h3_7ee527ff_tab` | -8.36 | 0 |  | Doc object properties | These properties define core document behavior: chapter numbering styles (numeric, Roman, Kanji, etc.), custom text for ... |
| 4 | `h2_d975e034_tab` | -8.54 | 0 |  | Constants | This chunk defines constants for footnote and change bar formatting in MIF. It includes numeric values controlling footn... |
| 5 | `h2_d567edc8_tab` | -9.12 | 0 |  | Constants | These constants define page numbering and numeric formatting styles in MIF, enabling precise control over document typog... |
| 6 | `h3_69b94320_tab` | -9.18 | 0 |  | BookComponent object properties | These properties control how a BookComponent behaves within a book structure: footnote and page numbering modes (continu... |
| 7 | `h2_a9976d5c_tab` | -9.28 | 0 |  | Constants | These constants define configurable properties for UI elements, primarily list and form controls, using integer IDs. Eac... |
| 8 | `h3_6a92524e` | -9.56 | 0 |  | SetProps | Function Summary > KeyCatalog > SetProps Sets key catalog properties via a PropVal list, aligning with AFrame’s SetProps... |
| 9 | `h3_a64326ee_tab` | -9.59 | 0 |  | Doc object properties | These properties control document section numbering behavior. SectionNumStyle defines the format (Arabic, Roman, alphabe... |
| 10 | `h2_43d20e69` | -9.60 | 0 |  | Changing Table Properties | The following script changes the left indentation of all the tables in a FrameMaker body page by 1 inch and the width of... |
| 11 | `h3_19763505_tab` | -9.61 | 0 |  | Doc object properties | These properties control document appearance, behavior, and state in FrameMaker. They manage formatting overrides, visib... |
| 12 | `h2_0e992ac0_tab` | -9.66 | 0 |  | Constants | These constants define property flags for frame and text block formatting in MIF, enabling precise control over position... |
| 13 | `h4_09495175_tab` | -9.78 | 0 |  | Ruling properties | <!-- Data Table --> MIF object,Description <TblColumnRuling (tagstring)>,Ruling style for most columns; value must match... |
| 14 | `h3_3417a124_tab` | -9.87 | 0 |  | Doc object properties | These properties control print behavior and formatting in non-fluid document views. They define start pages, numbering s... |
| 15 | `h3_2c1b0bf9` | -9.98 | 0 |  | GetProps | Function Summary > AFrame > GetProps GetProps() fetches all properties of a specified object without arguments, returnin... |

**Filter matches NOT in top 15** (47 chunks):

| # | Chunk ID | Heading | Content Preview |
|---|----------|---------|-----------------|
| 1 | `h2_e6bb15e6_tab` | Constants | These constants define essential page and paragraph properties for document layout and scripting in FrameMaker. They ena... |
| 2 | `h2_4e7c8409_tab` | Constants | These constants define paragraph formatting behaviors in a publishing system, controlling hyphenation, spacing, numberin... |
| 3 | `h3_3e1957a3` | Doc methods | AddNewBuildExpr, AddText, CenterOnText, Clear, ClearAllChangebars, Close, Compare, Copy, Cut, DeleteBuildExpr, DeleteTex... |
| 4 | `h3_fca0af9b_tab` | FmtChangeList object properties | The FmtChangeList properties define paragraph formatting controls, especially for Asian text spacing, punctuation rules ... |
| 5 | `h3_84cb1ead_tab` | Pgf object properties | These properties control paragraph-level typography and autonumbering in Asian text layouts. They define spacing rules f... |
| 6 | `h3_df6550fb_tab` | PgfFmt object properties | The PgfFmt properties define paragraph-level formatting behaviors for PDF output and Asian text handling. Key features i... |
| 7 | `h3_853895bf` | RestartPgfNumbering | Function Summary > Doc > RestartPgfNumbering Restarts paragraph numbering in the specified document, resetting counters ... |
| 8 | `h4_969690f3_tab` | Numbering properties | MIF Document Statements > Paragraph formats > Pgf statement > Numbering properties Enables automatic paragraph numbering... |
| 9 | `h1_78a77e9b` | Report document properties | This script demonstrates comprehensive document introspection by analyzing and reporting various properties of the activ... |
| 10 | `h1_019ab415_exa` | Report document properties | ```jsx var doc = app.ActiveDoc; if(doc.ObjectValid() == true) { // Display introductory information with document filena... |
| ... | (37 more) | | |

**YOUR PICK (write 1-3 chunk IDs):**
```
positive_ids: [h4_969690f3_tab]
```

---

## 29. q044: How can I auto-generate mini-TOCs for all chapters?

**Filters:**
  - Filter 1: fields=['chunk_summary', 'content', 'heading'], contains_any=['GenerateTOC', 'UpdateTOC', 'mini-TOC'], contains_all=[], max_matches=60
  - Filter 2: fields=['chunk_summary', 'content'], contains_any=['BookComponentIsGeneratable', 'BookComponentType', 'Chapter'], contains_all=[], max_matches=40

**Top 15 reranker-scored candidates** (filter match marked with [F]):

| Rank | Chunk ID | Score | Label | [F] | Heading | Content Preview |
|------|----------|-------|-------|-----|---------|-----------------|
| 1 | `h3_c95a93c2` | -1.16 | 1 | [F] | NewInlineComponentOfType | Creates an inline component. Presently only one type of inline component exists, and that is the mini-TOC. Returns: Void... |
| 2 | `h3_25c3ddea` | -4.70 | 1 | [F] | InlineComponentInfo statement | MIF Document Statements > Mini TOC > InlineComponentInfo statement The `InlineComponentInfo` statement configures a mini... |
| 3 | `h3_f5f72381_tab` | -4.85 | 1 | [F] | BookComponent object properties | These properties control how chapters are numbered, styled, and displayed in a book component. ChapterNumber and ChapNum... |
| 4 | `h2_05208145` | -6.31 | 0 |  | Mini TOC | FrameMaker document can contain a mini TOC. In a MIF file, a mini TOC tag is defined in an `InlineComponentsInfo` statem... |
| 5 | `h3_3c542fc3_tab` | -9.26 | 1 | [F] | BookComponent object properties | These properties define how a BookComponent is generated, typed, parented, and numbered. BookComponentIsGeneratable and ... |
| 6 | `h3_21049a1f` | -9.99 | 0 |  | Using the default layout | If you don't need to control the page layout of a document, you can use the default page layout by putting all of the do... |
| 7 | `h3_d726e4b7` | -10.25 | 1 | [F] | Creating a first master page | In addition to left and right master pages, you can create custom master page layouts that you can apply to body pages. ... |
| 8 | `h2_45a13e41` | -10.49 | 0 |  | Creating conditional text | Using MIF Statements > Creating conditional text You can generate multiple document variants from one source by tagging ... |
| 9 | `h4_2790dd80_tab` | -10.74 | 1 | [F] | Chapter numbering properties | <!-- Data Table --> MIF object,Description <ChapterNumStart (integer)>,Starting chapter number <ChapterNumStyle keyword>... |
| 10 | `h2_8a181675` | -10.94 | 0 |  | Pages | Pages in a MIF file are defined by a `Page` statement. A FrameMaker document can have four types of pages: * Body pages ... |
| 11 | `h4_c2e44eed` | -11.16 | 0 |  | Creating a simple page layout | If you want some control of the page layout but do not want to create master pages, you can use the `Document` substatem... |
| 12 | `h4_e65b0981` | -11.16 | 0 |  | Usage | Most MIF generators will put all document text in one `TextFlow` statement. However, if there are subsequent `TextFlow` ... |
| 13 | `h3_6122a9d9` | -11.16 | 0 |  | TextRect statement | The `TextRect` statement defines a text frame. It can appear at the top level or in a `Page` or `Frame` statement. <!-- ... |
| 14 | `h2_be9735b6` | -11.23 | 0 |  | Creating a simple MIF file for FrameMaker | The most accurate source of information about MIF files is a MIF file generated by FrameMaker. MIF files generated by Fr... |
| 15 | `h2_92d556cb_tab` | -11.23 | 1 | [F] | Constants | These constants define volume and chapter numbering behaviors in MIF documents. They control how numbers are computed (e... |

**Filter matches NOT in top 15** (11 chunks):

| # | Chunk ID | Heading | Content Preview |
|---|----------|---------|-----------------|
| 1 | `h2_5b3b7eb0` | InlineComponent | A component that can be placed inline within a Frammaker document. Currently FrameMaker supports the mini-TOC inline com... |
| 2 | `h1_818e46bb` | Differences Between JSX Scripts and the Framemaker... | Adobe FrameMaker scripts are modeled closely on the FrameMaker FDK. These scripts act as wrappers to the FDK and hide th... |
| 3 | `h2_55546c18_exa` | Verify hypertext marker links | ```jsx // Main entry point for hypertext link validation function checkHypertextLinks_Main() { var doc = app.ActiveDoc; ... |
| 4 | `h3_1d0a36c1_tab` | BookComponent object properties | The BookComponent properties define structural and formatting behaviors within a book hierarchy. VolumeNumStyle controls... |
| 5 | `h2_807caebf_tab` | Constants | These constants define numeric identifiers for dynamic document variables used in report generation, particularly for he... |
| 6 | `h2_4ce0670d_tab` | Constants | These constants define component navigation and book-level index types in a publishing system. Navigation flags (FP_) co... |
| 7 | `h2_d567edc8_tab` | Constants | These constants define page numbering and numeric formatting styles in MIF, enabling precise control over document typog... |
| 8 | `h3_7ee527ff_tab` | Doc object properties | These properties define core document behavior: chapter numbering styles (numeric, Roman, Kanji, etc.), custom text for ... |
| 9 | `h3_b82135d1_tab` | Doc object properties | These Doc object properties control PDF export behavior and chapter numbering in FrameMaker books. PDFStartPage, PDFZoom... |
| 10 | `h3_e8e803fb_tab` | FCodes object properties | These FCodes map keyboard commands to internal MIF operations, enabling text formatting, layout control, and document st... |
| ... | (1 more) | | |

**YOUR PICK (write 1-3 chunk IDs):**
```
positive_ids: [h3_25c3ddea, h2_5b3b7eb0]
```

---

## 30. q047: How do I rebuild cross-reference formats after importing from Word via MIF?

**Filters:**
  - Filter 1: fields=['chunk_summary', 'content'], contains_any=['XRefFmt', 'XRefSrcText', 'XRefIsUnresolved'], contains_all=[], max_matches=50
  - Filter 2: fields=['chunk_summary', 'content'], contains_any=['ImportFormats', 'Word import', 'doc.DeleteFmt'], contains_all=[], max_matches=40

**Top 15 reranker-scored candidates** (filter match marked with [F]):

| Rank | Chunk ID | Score | Label | [F] | Heading | Content Preview |
|------|----------|-------|-------|-----|---------|-----------------|
| 1 | `h3_b24ec725` | 3.03 | 1 | [F] | Inserting the reference point | The final step in creating a cross-reference is to insert an `XRef` statement at the position in text where the cross-re... |
| 2 | `h2_d1563e82` | 1.60 | 0 |  | Creating cross-references | In a FrameMaker document, you can create cross-references that are automatically updated. A cross-reference can refer to... |
| 3 | `h3_c379d146` | -0.55 | 1 | [F] | How FrameMaker writes cross-references | When FrameMaker writes a cross-reference, it provides the actual text that will appear at the reference point. This info... |
| 4 | `h3_f30c00eb` | -1.92 | 0 |  | Tips | The following hints may help you minimize the MIF statements for paragraph formats: * If possible, use the formats in th... |
| 5 | `h3_990893f8` | -2.10 | 1 | [F] | GetNamedXRefFmt | Function Summary > Doc > GetNamedXRefFmt Retrieves a named Cross Reference Format object by its identifier, enabling con... |
| 6 | `h3_e98fdb75` | -2.22 | 0 |  | XRefFormats and XRefFormat statements | The `XRefFormats` statement defines the formats of cross-references to be used in document text flows. A MIF file can ha... |
| 7 | `h3_845648d6` | -2.92 | 1 | [F] | NewNamedXRefFmt | Function Summary > Doc > NewNamedXRefFmt Creates a named Cross Reference Format for consistent document referencing. Ass... |
| 8 | `h3_321c2e0e` | -3.31 | 0 |  | NewAnchoredFormattedXRef | Function Summary > Doc > NewAnchoredFormattedXRef Creates an anchored, formatted cross-reference tied to a specific text... |
| 9 | `h3_75f8881c` | -3.90 | 0 |  | Creating cross-reference formats | The cross-reference formats for a document are defined in one `XRefFormats` statement. A document can have only one `XRe... |
| 10 | `h3_ffa3d1d6` | -3.90 | 0 |  | UpdateXRef | Updates the cross-references in a document. It performs the same operation as clicking Update in the Cross-Reference win... |
| 11 | `h3_a643ba9d` | -4.34 | 1 | [F] | SimpleImportFormats | Imports formats from a document to a document or a book. If you import formats to a book, the method imports formats to ... |
| 12 | `h2_107038ed_tab` | -4.55 | 1 | [F] | XRefFmt | MIF Object Reference > XRefFmt The XRefFmt table defines reusable cross-reference formats within a document. Each entry ... |
| 13 | `h3_3c4a4c55_tab` | -5.01 | 1 | [F] | SimpleImportFormats | This table maps binary flags to document format import options, enabling precise control over what elements are imported... |
| 14 | `h3_7f860344_exa` | -5.14 | 0 |  | Editing the MIF file | ```mif <Page <Unique 45155> <PageType BodyPage > <PageNum `1'> <PageSize 8.5" 11.0"> <PageOrientation Portrait > <PageAn... |
| 15 | `h4_3fc27316` | -5.41 | 0 |  | Import properties for importing Framemaker and MIF... | Import() uses the following properties only for importing FrameMaker documents and MIF files : <!-- Data Table --> Prope... |

**Filter matches NOT in top 15** (13 chunks):

| # | Chunk ID | Heading | Content Preview |
|---|----------|---------|-----------------|
| 1 | `h2_a42181a1_tab` | Constants | These constants define text formatting and document behavior options in a publishing system. They control line alignment... |
| 2 | `h2_11a19fea_tab` | Constants | These constants define internal identifiers for frame variables and formatting properties used in document processing, p... |
| 3 | `h2_3629a7b6_tab` | Constants | These constants define numeric identifiers for key document elements and states in FrameMaker’s API, enabling programmat... |
| 4 | `h2_01291df9_tab` | Constants | These constants define object types in the FrameMaker API, each mapped to a unique integer ID for programmatic identific... |
| 5 | `h3_0f3c0b00_tab` | Doc object properties | These properties expose key document elements for programmatic access, enabling navigation and manipulation of core comp... |
| 6 | `h3_3e1957a3` | Doc methods | AddNewBuildExpr, AddText, CenterOnText, Clear, ClearAllChangebars, Close, Compare, Copy, Cut, DeleteBuildExpr, DeleteTex... |
| 7 | `h2_be6149de_tab` | XRef | The XRef table defines properties for cross-references in FrameMaker, linking elements to their source and resolution st... |
| 8 | `h2_88705d5b_tab` | XRef | The XRef table defines cross-reference metadata linking source content to client-generated references. It distinguishes ... |
| 9 | `h3_ecc499f0` | GetNamedObject | Gets the object with the specified name and type. The method works with the following objects: - AttrCondExpr - CharFmt ... |
| 10 | `h3_de222528` | NewNamedObject | Creates the following named objects: - AttrCondExpr - CharFmt - CombinedFontDefn - Color - Command - CondFmt - ElementDe... |
| ... | (3 more) | | |

**YOUR PICK (write 1-3 chunk IDs):**
```
positive_ids: [Delete this question]
```

---

## 31. q053: What MIF objects control MathML control properties like font size and orientation?

**Filters:**
  - Filter 1: fields=['chunk_summary', 'content', 'heading'], contains_any=['MathML', 'MathML object properties', 'MathML methods'], contains_all=[], max_matches=60
  - Filter 2: fields=['chunk_summary', 'content'], contains_any=['NewMathML', 'Equation', 'MathML entity'], contains_all=[], max_matches=40

**Top 15 reranker-scored candidates** (filter match marked with [F]):

| Rank | Chunk ID | Score | Label | [F] | Heading | Content Preview |
|------|----------|-------|-------|-----|---------|-----------------|
| 1 | `h3_e8907d35` | -6.54 | 0 |  | app methods | MIF Object Reference > app > app methods These methods manage named MIF objects—books, commands, menus, separators, and ... |
| 2 | `h2_2c9e82c5` | -6.79 | 0 |  | Working with MIF files | Using MIF Statements > Working with MIF files MIF files offer a human-readable, ASCII version of FrameMaker documents, e... |
| 3 | `h4_782e215c_tab` | -7.54 | 0 |  | View only document properties | <!-- Data Table --> MIF object,Description <DViewOnly (boolean)>,Yes specifies View Only document (locked) <DViewOnlyXRe... |
| 4 | `h2_2d0830da` | -7.69 | 0 |  | Creating variables | In a FrameMaker document, variables act as placeholders for text that might change. For example, many documents use a va... |
| 5 | `h4_fd7aadce_tab` | -7.78 | 0 |  | Sub section numbering | <!-- Data Table --> MIF object,Description <SubSectionNumStart integer>,Starting Sub section number <SubSectionNumStyle ... |
| 6 | `h4_b51d5b8b` | -8.03 | 1 | [F] | Math properties | "For more information, see "MIF Equation Statements.", |
| 7 | `h1_7bcc29bc` | -8.04 | 0 |  | MIF Document Statements | Most MIF statements are listed in the order that they appear in a MIF file, as described in the following section. |
| 8 | `h3_681cc2fd_tab` | -8.46 | 1 | [F] | MathML object properties | These properties control MathML object positioning, sequencing, and rendering in a document. LocY defines vertical place... |
| 9 | `h3_7f860344_exa` | -8.47 | 0 |  | Editing the MIF file | ```mif <Page <Unique 45155> <PageType BodyPage > <PageNum `1'> <PageSize 8.5" 11.0"> <PageOrientation Portrait > <PageAn... |
| 10 | `h2_babfd83a` | -8.53 | 0 |  | Conditional text | MIF Document Statements > Conditional text MIF files manage conditional text via `Condition` statements that define visi... |
| 11 | `h3_24d16f44` | -8.59 | 1 | [F] | SetProps | Function Summary > MathML > SetProps Sets key properties on a MathML object via a PropVal list, enabling dynamic configu... |
| 12 | `h3_534cc11d_tab` | -8.74 | 1 | [F] | MathML object properties | These properties define visual and structural behaviors of MathML objects within a frame. Pen and TintPercent control ap... |
| 13 | `h3_c379d146` | -8.84 | 0 |  | How FrameMaker writes cross-references | When FrameMaker writes a cross-reference, it provides the actual text that will appear at the reference point. This info... |
| 14 | `h3_f94aa58d` | -8.92 | 1 | [F] | NewMathML | Function Summary > Doc > NewMathML Creates a MathML object as a child of a specified graphic frame, integrating mathemat... |
| 15 | `h3_3e1957a3` | -8.99 | 1 | [F] | Doc methods | AddNewBuildExpr, AddText, CenterOnText, Clear, ClearAllChangebars, Close, Compare, Copy, Cut, DeleteBuildExpr, DeleteTex... |

**Filter matches NOT in top 15** (30 chunks):

| # | Chunk ID | Heading | Content Preview |
|---|----------|---------|-----------------|
| 1 | `h2_30622ae7` | MathML | (empty) |
| 2 | `h3_b85c5721_tab` | MathML object properties | These properties control MathML rendering and visual behavior: display/compose DPI, font size, XML data, and transformat... |
| 3 | `h3_e4d75e1a_tab` | MathML object properties | These properties define visual and interactive behaviors of MathML objects within a layout system. They control position... |
| 4 | `h3_8db32867_tab` | MathML object properties | These properties define visual and behavioral attributes of MathML graphic objects, controlling appearance (arrow styles... |
| 5 | `h3_1544fb9c` | MathML methods | Delete, GetProps, SetProps, ObjectValid. |
| 6 | `h2_a5f6c816` | MathML | (empty) |
| 7 | `h3_d3aa2247_tab` | Style statement | <ArrowStyle ...>, <RunaroundGap (dimension)>,Space between the object and the text flowing around the object; must be a ... |
| 8 | `h2_be261696` | Size graphic to fit frame | This script demonstrates intelligent graphic scaling by resizing graphics to fit within their parent anchored frames whi... |
| 9 | `h2_b90c985e_tab` | Constants | These constants define numeric identifiers for file types and configuration properties in a document processing system. ... |
| 10 | `h2_14969c63_tab` | Constants | These constants define object types and data types used in MIF processing. Object constants (FV_FO_*) identify structure... |
| ... | (20 more) | | |

**YOUR PICK (write 1-3 chunk IDs):**
```
positive_ids: [h3_b85c5721_tab]
```

---

## 32. q055: How can I enable AutoBackup/AutoSave checkpoints via ExtendScript during long conversions?

**Filters:**
  - Filter 1: fields=['chunk_summary', 'content'], contains_any=['FS_AutoBackupOnSave', 'FV_SaveYesAutoBackup', 'AutoBackup'], contains_all=[], max_matches=40
  - Filter 2: fields=['chunk_summary', 'content'], contains_any=['DocSaveType', 'SaveFmt', 'SimpleSave'], contains_all=[], max_matches=40

**Top 15 reranker-scored candidates** (filter match marked with [F]):

| Rank | Chunk ID | Score | Label | [F] | Heading | Content Preview |
|------|----------|-------|-------|-----|---------|-----------------|
| 1 | `h1_ce66bbd2` | -8.13 | 0 |  | JSX example scripts | ExtendScript (JSX) is similar to JavaScript. You can easily develop ExtendScript for any of the applications in FrameMak... |
| 2 | `h2_ef9b4c3f` | -9.42 | 0 |  | Notifications | Notifications is the internal mechanism through which a script registered for a particular event is run when the event i... |
| 3 | `h2_bdee190a` | -9.55 | 0 |  | Notify on document open and close | This script demonstrates event-driven programming by establishing automatic response handlers for document lifecycle eve... |
| 4 | `h2_d1563e82` | -9.72 | 0 |  | Creating cross-references | In a FrameMaker document, you can create cross-references that are automatically updated. A cross-reference can refer to... |
| 5 | `h2_96cc0c95_exa` | -10.02 | 0 |  | Table selection and sizing | ```jsx var doc = app.ActiveDoc; // Get active document from FrameMaker session if(doc.ObjectValid() == true) { // Valida... |
| 6 | `h2_1545d309` | -10.02 | 0 |  | Get active document | This script demonstrates fundamental document object handling in FrameMaker ExtendScript by checking for an active docum... |
| 7 | `h4_3fc27316` | -10.33 | 0 |  | Import properties for importing Framemaker and MIF... | Import() uses the following properties only for importing FrameMaker documents and MIF files : <!-- Data Table --> Prope... |
| 8 | `h2_5cc7fec6` | -10.35 | 0 |  | Adding Text and Enabling Change Bars | JSX example scripts > Adding Text and Enabling Change Bars The script inserts “Hello” at the start of the first paragrap... |
| 9 | `h2_f02aee80_exa` | -10.41 | 0 |  | Book operations: navigation, opening, updating | ```jsx // Robust file opening function with comprehensive error handling function OpenFile(path) { var props = GetOpenDe... |
| 10 | `h3_210d8705` | -10.48 | 0 |  | QuickSelect | Implements a quick-key interface that allows the user to choose a string from a list of strings in the docu -ment Tag ar... |
| 11 | `h4_036d3a4b_tab` | -10.56 | 0 |  | Reference properties | MIF Document Statements > Global document properties > Document statement > Reference properties These properties enable... |
| 12 | `h2_1ea8fe93_exa` | -10.69 | 0 |  | Creating formatting shortcuts | ```jsx // Create and display command catalog dialog function LaunchCommandCatalog() { var res = """palette { properties:... |
| 13 | `h2_74247b01_exa` | -10.73 | 0 |  | Creating a graphics utilities palette/menu | ```jsx // Function to resize graphics with expansion/shrinkage control function SizeGraphic(graphic, height_inc, width_i... |
| 14 | `h2_5e3e9ffa_exa` | -10.79 | 0 |  | Dialogs basic palette operation with coordinates | ```jsx // Function to resize graphics with expansion/shrinkage control function SizeGraphic(graphic, height_inc, width_i... |
| 15 | `h2_1e257d0f_exa` | -10.86 | 0 |  | Creating formatting shortcuts | ```jsx // Capture formatting state before character format application for undo function CaptureChrFormatUndoSnapshot(do... |

**Filter matches NOT in top 15** (14 chunks):

| # | Chunk ID | Heading | Content Preview |
|---|----------|---------|-----------------|
| 1 | `h2_2c09c6a3_tab` | Constants | These constants provide system and environment metadata for FileMaker Pro scripting, enabling scripts to query runtime p... |
| 2 | `h2_632c8fcb_tab` | Constants | These constants define configuration flags for saving operations in MIF, controlling auto-backup behavior, page count ru... |
| 3 | `h2_472abbac_tab` | Constants | These constants define file save behaviors and formats in a versioned application system. They distinguish between binar... |
| 4 | `h3_4bce7caf_tab` | app object properties | The app object properties define core session-wide state and configuration for FrameMaker, including active documents, v... |
| 5 | `h3_c12daad3` | Book methods | MIF Object Reference > Book > Book methods These methods manage book lifecycle and structure: create, import, save, and ... |
| 6 | `h2_130c6bdf_tab` | Constants | These constants define document and UI behavior settings in a publishing system, primarily controlling formatting, visib... |
| 7 | `h2_e5bb510b_tab` | Constants | These constants define file save formats for FrameMaker, mapping integer values to specific binary or interchange format... |
| 8 | `h2_ad9657b6_tab` | Constants | These constants define error codes and file format identifiers for file operations in the MIF system. Error constants (e... |
| 9 | `h3_a219b05a_tab` | Doc object properties | These properties control document behavior during save and open operations. DocSaveType defines the output format (binar... |
| 10 | `h3_3e1957a3` | Doc methods | AddNewBuildExpr, AddText, CenterOnText, Clear, ClearAllChangebars, Close, Compare, Copy, Cut, DeleteBuildExpr, DeleteTex... |
| ... | (4 more) | | |

**YOUR PICK (write 1-3 chunk IDs):**
```
positive_ids: [h3_4bce7caf_tab]
```

---

## 33. q057: How can I update a book and TOC after deleting chapters to avoid orphan entries?

**Filters:**
  - Filter 1: fields=['chunk_summary', 'content'], contains_any=['GenerateTOC', 'BookComponentIsGeneratable', 'BookComponentType'], contains_all=[], max_matches=50
  - Filter 2: fields=['chunk_summary', 'content'], contains_any=['UpdateBook', 'BookComponentStatus', 'ChapNumComputeMethod'], contains_all=[], max_matches=40

**Top 15 reranker-scored candidates** (filter match marked with [F]):

| Rank | Chunk ID | Score | Label | [F] | Heading | Content Preview |
|------|----------|-------|-------|-----|---------|-----------------|
| 1 | `h3_ea62e5d2` | -10.09 | 0 |  | ManageConditionalExpressions | Add, edit, or delete conditional expression tags to the current book. Applies to the options available in the Add/Edit C... |
| 2 | `h3_88d2b9c9` | -10.37 | 0 |  | Delete | Deletes the specified BookComponent object. See Delete under the AFrame class for more information. Returns: int Syntax:... |
| 3 | `h3_f30b8b09` | -10.67 | 0 |  | NewSeriesBookComponent | Function Summary > Book > NewSeriesBookComponent NewSeriesBookComponent() inserts a new Book Component into a series at ... |
| 4 | `h3_dcda5882` | -10.68 | 0 |  | SimpleSave | The SimpleSave() method saves a book. If you set the interactive parameter to False and you specify the book's current n... |
| 5 | `h3_d44c43cd` | -10.77 | 1 | [F] | UpdateBook | The UpdateBook() method updates a book. The method allows you to specify a script (property list) specifying how to upda... |
| 6 | `h3_f102f803` | -10.91 | 0 |  | MoveComponent | Function Summary > BookComponent > MoveComponent Moves a book component up/down in sequence or promotes/demotes it withi... |
| 7 | `h3_3c542fc3_tab` | -10.93 | 1 | [F] | BookComponent object properties | These properties define how a BookComponent is generated, typed, parented, and numbered. BookComponentIsGeneratable and ... |
| 8 | `h3_5f87d866` | -10.96 | 0 |  | Delete | Deletes a reference page. See Delete under the AFrame class for more information. Returns: int Syntax: Delete() |
| 9 | `h2_55546c18_exa` | -11.04 | 0 |  | Verify hypertext marker links | ```jsx // Main entry point for hypertext link validation function checkHypertextLinks_Main() { var doc = app.ActiveDoc; ... |
| 10 | `h3_c95a93c2` | -11.08 | 0 |  | NewInlineComponentOfType | Creates an inline component. Presently only one type of inline component exists, and that is the mini-TOC. Returns: Void... |
| 11 | `h3_87484946_tab` | -11.12 | 0 |  | Pgf object properties | These properties define paragraph formatting and behavior in MIF, controlling layout, spacing, alignment, and document i... |
| 12 | `h3_b64696d3` | -11.12 | 0 |  | SimpleSave | Saves a document or book. If you set the interactive parameter to False and specify the document or book's current name ... |
| 13 | `h3_78e48d7d` | -11.13 | 0 |  | Delete | Deletes a footnote. See Delete under the AFrame class for more information. Returns: int Syntax: Delete() |
| 14 | `h3_ccb731e6` | -11.14 | 0 |  | Reformat | Function Summary > Doc > Reformat Reformat() applies formatting to documents in the current session after re-enabling re... |
| 15 | `h3_b3215d23` | -11.15 | 0 |  | NewBookComponentOfTypeInHierarchy | Function Summary > Book > NewBookComponentOfTypeInHierarchy Adds a structured book component of a specified type at a pr... |

**Filter matches NOT in top 15** (9 chunks):

| # | Chunk ID | Heading | Content Preview |
|---|----------|---------|-----------------|
| 1 | `h2_4ce0670d_tab` | Constants | These constants define component navigation and book-level index types in a publishing system. Navigation flags (FP_) co... |
| 2 | `h2_0bf04d3e` | Book operations: navigation, opening, updating | This script demonstrates comprehensive book management by navigating book components, opening files with robust error ha... |
| 3 | `h3_f5f72381_tab` | BookComponent object properties | These properties control how chapters are numbered, styled, and displayed in a book component. ChapterNumber and ChapNum... |
| 4 | `h2_5beeb826_tab` | Constants | These constants define error codes and layout positions for book management and document structure in MIF. Error codes (... |
| 5 | `h2_404e7d8a_tab` | Constants | These constants define configuration flags and status codes for book management in a publishing system. Flags (FS_*) con... |
| 6 | `h3_b82135d1_tab` | Doc object properties | These Doc object properties control PDF export behavior and chapter numbering in FrameMaker books. PDFStartPage, PDFZoom... |
| 7 | `h3_872d5a12_tab` | UpdateBook | The UpdateBook function orchestrates targeted book-wide updates via configurable flags, controlling whether error logs d... |
| 8 | `h3_6d92cb7a_tab` | UpdateBook | The UpdateBook function configures how FrameMaker handles book updates under edge conditions. It controls user notificat... |
| 9 | `h3_fc9cb5fd` | UpdateBook (CheckStatus constant check) | To determine if a particular Constants.FS\_UpdateBookStatus (1) bit is set, use CheckStatus(). The method returns FE\_Su... |

**YOUR PICK (write 1-3 chunk IDs):**
```
positive_ids: [h3_6186c3fe]
```

---

## 34. q058: How do I enforce consistent footnote numbering across files merged into one book?

**Filters:**
  - Filter 1: fields=['chunk_summary', 'content'], contains_any=['DFNoteRestart', 'BFNoteComputeMethod', 'BFNoteNumStyle'], contains_all=[], max_matches=50
  - Filter 2: fields=['chunk_summary', 'content'], contains_any=['Footnote numbering', 'BFNoteStartNum', 'BFNoteRestart'], contains_all=[], max_matches=40

**Top 15 reranker-scored candidates** (filter match marked with [F]):

| Rank | Chunk ID | Score | Label | [F] | Heading | Content Preview |
|------|----------|-------|-------|-----|---------|-----------------|
| 1 | `h3_44f6bd5a` | -0.83 | 0 |  | InitialAutoNums statement | The `InitialAutoNums` statement controls the starting values for autonumber series in a document. A MIF file can have on... |
| 2 | `h3_bf9191a6_tab` | -1.97 | 0 |  | BookComponent object properties | These properties define how a BookComponent behaves within a document book structure. They control page numbering styles... |
| 3 | `h3_853895bf` | -4.70 | 0 |  | RestartPgfNumbering | Function Summary > Doc > RestartPgfNumbering Restarts paragraph numbering in the specified document, resetting counters ... |
| 4 | `h3_69b94320_tab` | -5.68 | 1 | [F] | BookComponent object properties | These properties control how a BookComponent behaves within a book structure: footnote and page numbering modes (continu... |
| 5 | `h2_de0341af_tab` | -5.81 | 1 | [F] | Constants | These constants define formatting rules for footnote numbering and reference behavior in MIF. Numeric styles (0x03–0x10)... |
| 6 | `h3_abfbaf11_tab` | -6.03 | 1 | [F] | Doc object properties | These properties configure footnote formatting in a document: prefix/suffix wrap the footnote number, NumComputeMethod c... |
| 7 | `h3_5f7acaec_tab` | -6.89 | 1 | [F] | BookComponent object properties | These properties control footnote and volume numbering behaviors in book components. TblFnNumStyle defines table footnot... |
| 8 | `h3_a643ba9d` | -7.12 | 0 |  | SimpleImportFormats | Imports formats from a document to a document or a book. If you import formats to a book, the method imports formats to ... |
| 9 | `h3_c0f747cf` | -7.18 | 0 |  | Notes statement | The Notes statement defines all of the footnotes that will be used in a table title, cell, or text flow. It can appear a... |
| 10 | `h3_cbdda16a` | -7.18 | 0 |  | ApplyPageLayout | The ApplyPageLayout() method applies the layout of one page to another page. The method returns FE\_Success on success. ... |
| 11 | `h3_a1fca523_tab` | -7.53 | 1 | [F] | BookComponent object properties | These properties control page and footnote numbering behavior within a BookComponent. FirstPageNum restarts page numberi... |
| 12 | `h4_2790dd80_tab` | -7.59 | 0 |  | Chapter numbering properties | <!-- Data Table --> MIF object,Description <ChapterNumStart (integer)>,Starting chapter number <ChapterNumStyle keyword>... |
| 13 | `h3_845648d6` | -7.60 | 0 |  | NewNamedXRefFmt | Function Summary > Doc > NewNamedXRefFmt Creates a named Cross Reference Format for consistent document referencing. Ass... |
| 14 | `h3_3945da91_tab` | -7.61 | 0 |  | Doc object properties | The Doc object properties define page layout behavior: PageNumStyle selects numbering formats (Arabic, Roman, Kanji, etc... |
| 15 | `h2_d1563e82` | -7.75 | 0 |  | Creating cross-references | In a FrameMaker document, you can create cross-references that are automatically updated. A cross-reference can refer to... |

**Filter matches NOT in top 15** (3 chunks):

| # | Chunk ID | Heading | Content Preview |
|---|----------|---------|-----------------|
| 1 | `h4_20778869_tab` | Footnote properties | <!-- Data Table --> MIF object,Description <DFNoteTag (string)>,Paragraph and reference frame tag for document footnotes... |
| 2 | `h2_b0f692dc_tab` | Constants | These constants define unique integer identifiers for frame, footnote, marker, and variable properties in FramerScript. ... |
| 3 | `h3_5bfc3b99_tab` | Doc object properties | These properties control footnote formatting specifically for tables within a document. TblFnNumStyle defines how footno... |

**YOUR PICK (write 1-3 chunk IDs):**
```
positive_ids: [h4_20778869_tab, h4_81ba3866_tab]
```

---

## 35. q173: What is an ID in MIF and what range is valid?

**Filters:**
  - Filter 1: fields=['content', 'chunk_summary'], contains_any=['ID can be any positive integer', 'An ID can be any positive integer from 1 to 65535'], contains_all=[], max_matches=50

**Top 15 reranker-scored candidates** (filter match marked with [F]):

| Rank | Chunk ID | Score | Label | [F] | Heading | Content Preview |
|------|----------|-------|-------|-----|---------|-----------------|
| 1 | `h4_97c7f937_tab` | 0.81 | 0 |  | PDF Document Info | MIF object,Description <Document,Document properties <DNextUnique (ID)>,Refers to the next object with a <Unique `ID`> s... |
| 2 | `h4_1a3e0a91_tab` | -1.72 | 0 |  | Document properties | <!-- Data Table --> MIF object,Description <DNextUnique (ID)>,Refers to the next object with a <Unique ID> statement; ge... |
| 3 | `h4_e0c3ac97` | -2.60 | 0 |  | Usage | MIF Document Statements > Graphic objects and graphic frames > Generic object statements > Usage An object needs an `ID`... |
| 4 | `h4_8825749c` | -4.10 | 1 | [F] | About ID numbers | Using MIF Statements > Creating and applying character formats > Adding a table anchor > About ID numbers The `ATbl` sta... |
| 5 | `h3_5e29a849_tab` | -4.17 | 1 | [F] | MIF data items | This term or symbol,Means string,"Left quotation mark (`), zero or more standard ASCII characters (you can also include ... |
| 6 | `h3_a7a36834_tab` | -4.43 | 0 |  | FCodes object properties | The FCodes object defines system-level command identifiers for UI and input control in MIF applications. These constants... |
| 7 | `h2_7a71ea10_tab` | -4.47 | 0 |  | Constants | These constants define a sequential enum-like range for object types in the MIF system, starting from FO_Num (79). The v... |
| 8 | `h2_5c2e2a35_tab` | -4.51 | 0 |  | Constants | These constants define system-wide identifiers for XML, server, and book-state properties in MIF. They enable precise co... |
| 9 | `h2_67c6ac89_tab` | -4.79 | 0 |  | Constants | These constants define core document manipulation and formatting behaviors in Framer’s MIF system. They enable programma... |
| 10 | `h2_73f4a0b2_tab` | -4.79 | 0 |  | Constants | These constants define standardized color values and a family name identifier for MIF objects, enabling consistent color... |
| 11 | `h2_3c9577d0_tab` | -5.03 | 0 |  | Constants | These constants define integer identifiers for formatting and object types within the MIF system, enabling precise type ... |
| 12 | `h2_79674928_tab` | -5.71 | 0 |  | Constants | These constants define error and status codes for hypertext link resolution in MIF, covering missing/invalid arguments, ... |
| 13 | `h2_03349686_tab` | -5.76 | 0 |  | Constants | These constants define core formatting and rule-related identifiers in the MIF system, enabling precise control over doc... |
| 14 | `h2_caf2caa1_tab` | -5.80 | 0 |  | Constants | These constants define standardized identifiers for data types and plugin metadata in MIF systems. FT_ prefixes map to s... |
| 15 | `h2_ad11e077` | -5.85 | 0 |  | How FrameMaker identifies MIF files | MIF overview > How FrameMaker identifies MIF files FrameMaker identifies MIF files by the presence of a `MIFFile` or `Bo... |

**YOUR PICK (write 1-3 chunk IDs):**
```
positive_ids: [delete this question]
```

---
