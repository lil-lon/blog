# Introduction of Docling and DocLang

## Have you ever fought with Claude Code over a PDF?

Have you ever handed a PDF to an AI assistant and gotten back answers that were subtly off? The content is perfectly readable to you, yet the model keeps getting it wrong.

This is not a model intelligence problem. It is a format problem.

<!-- TODO: confirm how Claude Code handles PDFs internally. The linked page documents the Claude API, not Claude Code. https://platform.claude.com/docs/en/build-with-claude/pdf-support -->
Modern assistants do have machinery for ingesting PDFs. They typically combine extracted text with rendered page images so a vision-capable model can "look" at the page. This works, but it inherits every weakness of the PDF format itself.

## PDF was never meant for machines

A PDF describes where to place glyphs on a page so that ink lands in the right spot when printed. That is the entire job it was designed for. It does not record that a run of characters is a heading, that twelve numbers form a 3x4 table, or that "Name:" is followed by a fillable field. The PDF backend used by the tools below says so in its own documentation: the library "does not implement layout analysis, such as detecting words, lines, or paragraphs."

So when an AI reads a PDF, it has to reverse-engineer structure that the format threw away:

- **Tables** collapse into a flat stream of numbers with no row or column boundaries.
- **Multi-column layouts** interleave the left and right columns line by line.
- **Formulas** turn into garbled symbol soup.
- **Scanned pages** carry no text layer at all.
- **Provenance** (which page and where on it) is lost the moment you extract the text.

Markdown, the format most AI pipelines fall back to, fixes some of this. It has headings, lists, and simple tables, and it tokenizes cleanly. But it is lossy in the other direction. It cannot record geometry, it cannot express merged or nested table cells, it has no notion of a form field, a background layer, or a cross-reference. Markdown is a good destination for prose and a poor one for real-world documents.

## Two open-source projects built for this gap

The document-AI gap is now being closed in the open. Two projects, both governed under the [LF AI & Data Foundation](https://lfaidata.foundation/projects/), tackle the two halves of the problem:

- **Docling** is the conversion tool. It turns PDFs, DOCX, PPTX, XLSX, HTML, images, and more into a structured representation. It has grown past 61k GitHub stars and is widely used as a document-ingestion front end for RAG and agentic pipelines.
- **DocLang** is the format specification. It defines an open, AI-native markup for representing a document's structure, semantics, layout, and geometry in one unambiguous form.

In short: **Docling is the tool, DocLang is the standard.** The rest of this post explains each one and why the pairing matters.

## Docling: parse once, export many ways

Docling's core idea is to parse a document a single time into a rich in-memory model called `DoclingDocument`, then serialize that model into whatever a downstream consumer needs.

The parsing does not rely on brittle OCR-first heuristics. Docling uses purpose-built models: a layout model that detects page elements (text blocks, headings, tables, figures, captions) and a dedicated table-structure model (TableFormer) that recovers the row and column grid of a table from its image. The result is a single object that knows the document's reading order, element types, and the page and bounding box of every piece of content.

From that one object you can export to several formats:

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
