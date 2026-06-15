# Introduction of Docling and DocLang

## Have you ever fought with Claude Code over a document?

Have you ever handed a document to an AI assistant and gotten back answers that were subtly off? A PDF spec, a Word report, a research paper, a scanned form. The content is perfectly readable to you, yet the model keeps getting it wrong.

This is not a model intelligence problem. It is a format problem.

## How Claude already deals with PDFs

Claude is not helpless with PDFs. The Claude API documents first-class [PDF support](https://platform.claude.com/docs/en/build-with-claude/pdf-support), handling both the extracted text and the visual layout of each page. Claude Code does not document a PDF mechanism of its own, but Anthropic ships an official [PDF skill](https://github.com/anthropics/skills/tree/main/skills/pdf) that an agent can load on demand.

That skill is a toolbox of conventional Python libraries. It uses `pdfplumber` for layout-aware text and table extraction, `pypdf` for structural operations such as merge, split, rotate, and encrypt, and `pytesseract` with `pdf2image` for OCR on scanned pages. To fill an interactive form it renders the page to an image and checks field bounding boxes so values land in the right place.

For one-off jobs (fill this form, merge these reports, pull the text out of a memo) this is plenty.

## Why the skill is not the whole answer

The skill works because most PDFs cooperate. It stops working where the format does. A PDF only records where to place glyphs on a page so the ink lands correctly when printed. It does not record that a line is a heading, that a block of numbers is a 3x4 table, or that "Name:" labels a fillable field. The PDF backend behind these tools says so in its own documentation: it "does not implement layout analysis, such as detecting words, lines, or paragraphs."

So the skill has to reverse-engineer structure the format threw away, and the seams show:

- **Tables.** `pdfplumber` recovers a grid from ruling lines or text alignment. Give it a borderless table, merged cells, or nested cells, and the rows and columns come out wrong.
- **Reading order.** A two-column page is just glyphs at coordinates. Naive extraction interleaves the columns line by line.
- **Scanned pages.** With no text layer, you fall back to `pytesseract`, the slow and error-prone OCR that document AI has spent years trying to escape.
- **Formulas and code** degrade into symbol soup, with no LaTeX and no language tag.
- **Provenance and semantics stay separate.** You can get text, and separately some bounding boxes, but nothing hands you one object that says "this is a heading, on page 3, at this box, in the body layer."

The deeper problem is that the output is ad hoc. Each task produces a string or a DataFrame for that task, with no shared, checkable representation behind it. There is nothing to validate and nothing another system can rely on as an interchange format. That is fine for a quick fix and wrong for a pipeline.

Markdown, the format most AI pipelines fall back to, only goes halfway. It has headings, lists, and simple tables, and it tokenizes cleanly. But it is lossy in the other direction. It cannot record geometry, it cannot express merged or nested table cells, and it has no notion of a form field, a background layer, or a cross-reference. Markdown is a good destination for prose and a poor one for real-world documents.

## Two open-source projects built for this gap

The document-AI gap is now being closed in the open. Two projects, both governed under the [LF AI & Data Foundation](https://lfaidata.foundation/projects/), tackle the two halves of the problem:

- **Docling** is the conversion tool. It turns PDFs, DOCX, PPTX, XLSX, HTML, images, and more into a structured representation. It has grown past 61k GitHub stars and is widely used as a document-ingestion front end for RAG and agentic pipelines.
- **DocLang** is the format specification. It defines an open, AI-native markup for representing a document's structure, semantics, layout, and geometry in one unambiguous form.

In short: **Docling is the tool, DocLang is the standard.** The rest of this post explains each one and why the pairing matters.

## Docling: many formats in, one model out

Real document piles are a mix of Word files, slide decks, spreadsheets, web pages, LaTeX sources, scans, and more, each with its own quirks. Docling's real strength is the funnel: it ingests PDF, DOCX, PPTX, XLSX, HTML, Markdown, AsciiDoc, LaTeX, images, and audio, and turns every one of them into a single in-memory model called `DoclingDocument`. The messy variety goes in, one uniform representation comes out.

How much work that takes depends on the input. Born-digital formats like DOCX or LaTeX already carry their structure, so it is mostly a matter of reading it out. PDFs and scans are where Docling earns its keep: it does not rely on brittle OCR-first heuristics but on purpose-built models, a layout model that detects page elements (headings, tables, figures, captions) and a table-structure model (TableFormer) that recovers a table's grid from its image. Either way you end up with the same object, one that knows the document's reading order, element types, and the page and bounding box of every piece of content.

From that one model you export to whatever a downstream consumer needs:

```python
from docling.document_converter import DocumentConverter

doc = DocumentConverter().convert("paper.pdf").document

doc.export_to_markdown()   # human-friendly, great for prose
doc.export_to_doctags()    # compact tag stream used to train vision-language models
doc.export_to_doclang()    # the AI-native standard format
```

That last line is the bridge to the second half of the story.

## DocLang: a format designed for how LLMs actually read

DocLang is an XML-based format. Its design goal is stated plainly in the spec: map cleanly to LLM tokens while preserving structure, semantics, layout, and geometry in a single representation. It distills lessons from two earlier innovations, OTSL (a compact table-structure language) and DocTags (a structure-preserving tag format), into one standard.

A few features show why it is more AI-friendly than PDF or Markdown.

**Semantic elements, not visual placement.** DocLang has first-class elements for the things a document is actually made of: `heading`, `text`, `list`, `table`, `picture`, `formula`, `code`, `footnote`, `page_header`, `page_footer`, and form constructs like `key`, `value`, and `field_region`. An AI does not have to guess what a block is, because the format already says so.

**Geometry travels with content.** Every element can carry an `element_head` that attaches metadata: a `location` (a bounding box quantized to a resolution grid, the same idea a vision model uses for location tokens), a `layer` (body, background, or furniture such as headers and footers), a `label`, and references like `thread` and `xref` that link related elements across the document. This is exactly the provenance that Markdown throws away, which means an answer can point back to the page and region it came from.

**Tables that survive contact with reality.** DocLang represents tables with OTSL-style tokens (`fcel` for a filled cell, `ched` for a column header, `lcel` and `ucel` for cells merged left or up, `nl` for a new row). Merged cells, header rows, and nested structure all survive. A Markdown table cannot express any of that.

**Forms have meaning.** A fillable field is not just the text "Name:". It is a `field_item` containing a `key` and a `value` marked `fillable`. The semantics that a human infers visually are written down explicitly.

**It is verifiable.** DocLang ships a reference validator that checks a document against an XSD schema (structure) and Schematron rules (constraints the schema cannot express, such as "every table row has the same number of cells"). A format you can validate is a format you can build reliable systems on.

Put together, the pipeline is: **Docling parses the messy real-world document, and DocLang is the clean, checkable, AI-native form it hands to the model.**

## A concrete example: converting a paper from PDF to DocLang

<!-- TODO: convert a real paper PDF with Docling and export to both Markdown and DocLang.
     Show a table and the reading order that naive PDF text extraction breaks but Docling
     keeps, point out the OTSL table tokens and location data preserved in the DocLang output,
     then validate that output with the DocLang reference validator. -->
*TODO: concrete code example (PDF to DocLang via Docling).*

## Takeaway

Documents are still one of the richest sources of enterprise and scientific knowledge, and almost none of them were designed for machines to read. Docling and DocLang attack that mismatch from both ends: a robust converter that understands real documents, and an open standard that records what it understood in a form built for language models. Together they are turning into the open document-AI stack worth knowing about.
