---
name: drawio
description: Use when creating or editing draw.io diagrams — generates mxGraphModel XML, exports to JPG/SVG/PDF, and visually reviews the result for elegance and correctness, iterating until the diagram fits its purpose
---

# Draw.io Diagram Skill

Generate draw.io diagrams as native `.drawio` files with visual review. Every diagram is exported to JPG and visually inspected, iterating until it is elegant, readable, and correct.

**IMPORTANT: Always use JPG (not PNG) for the internal review loop.** Claude Code gets API Error 400 when reading PNG files exported by draw.io. JPG works reliably. The user can later re-export from the `.drawio` source to any format they need.

## Core Workflow

1. **Generate XML** in mxGraphModel format for the requested diagram
2. **Write the XML** to a `.drawio` file using the Write tool
3. **Export to JPG** using the draw.io CLI (`-f jpg`)
4. **Visually review** — use the Read tool on the JPG to inspect the rendered diagram
5. **Iterate** — if anything is off (overlaps, bad spacing, unclear flow, ugly styling), edit the `.drawio` file and re-export. Repeat until satisfied.
6. **Deliver** — report the final result to the user. If they requested a specific format (SVG, PDF), export to that format too.

## Visual Review Checklist

After each export, inspect the JPG and check:

- **Layout**: Logical flow direction (top-to-bottom or left-to-right), no crossing arrows where avoidable
- **Spacing**: Elements evenly distributed, adequate whitespace, no cramped labels
- **Overlaps**: No overlapping nodes, labels, or edges
- **Connections**: All arrows connect to the correct source/target, edge labels are readable
- **Readability**: Font sizes large enough, labels concise, consistent terminology
- **Consistency**: Uniform colors, shapes, and styles for elements of the same type
- **Proportions**: Node sizes proportional to content, diagram fits well in the canvas
- **Elegance**: Clean, professional appearance — would you put this in a presentation?

If any check fails, fix the XML and re-export. There is no limit on iterations.

## Output Format

Check the user's request for a format preference:

- `create a flowchart` → `flowchart.drawio` + `flowchart.drawio.jpg` (for review)
- `svg: ER diagram` → `er-diagram.drawio` + `er-diagram.drawio.jpg` (for review) + `er-diagram.drawio.svg` (deliverable)
- `pdf architecture overview` → `architecture-overview.drawio` + `architecture-overview.drawio.jpg` (for review) + `architecture-overview.drawio.pdf` (deliverable)

Always export a JPG for visual review. The user can re-export from the `.drawio` source to any format they need.

### Supported export formats

| Format | Embed XML | Notes |
|--------|-----------|-------|
| `jpg`  | No        | **Use for visual review** — Claude Code reads JPG reliably |
| `png`  | Yes (`-e`) | DO NOT use for review (causes API Error 400 in Claude Code) |
| `svg`  | Yes (`-e`) | Scalable, editable in draw.io |
| `pdf`  | Yes (`-e`) | Printable, editable in draw.io |

PNG, SVG, and PDF support `--embed-diagram` — the exported file contains the full diagram XML, so opening it in draw.io recovers the editable diagram. JPG does not support embedding.

## draw.io CLI

### Locating the CLI

Try `drawio` first (works if on PATH), then fall back to:

- **macOS**: `/Applications/draw.io.app/Contents/MacOS/draw.io`

Use `which drawio` to check if it's on PATH before falling back.

### Export command

For visual review (JPG — no embed support):
```bash
drawio -x -f jpg -b 10 -o <output.drawio.jpg> <input.drawio>
```

For user-requested formats (with embedded diagram XML):
```bash
drawio -x -f <svg|pdf> -e -b 10 -o <output> <input.drawio>
```

Key flags:
- `-x` / `--export`: export mode
- `-f` / `--format`: output format (jpg, svg, pdf)
- `-e` / `--embed-diagram`: embed diagram XML in output (not supported for jpg)
- `-o` / `--output`: output file path
- `-b` / `--border`: border width around diagram (default: 0)
- `-s` / `--scale`: scale the diagram size
- `--width` / `--height`: fit into specified dimensions (preserves aspect ratio)

### Opening the result

```bash
open <file>
```

## File Naming

- Descriptive filename based on diagram content (e.g., `login-flow`, `database-schema`)
- Lowercase with hyphens for multi-word names
- Double extensions for exports: `name.drawio.jpg`, `name.drawio.svg`
- Keep the `.drawio` source file — it is the editable original

## XML Format

A `.drawio` file is native mxGraphModel XML. Always generate XML directly.

### Basic structure

Every diagram must have this structure:

```xml
<mxGraphModel>
  <root>
    <mxCell id="0"/>
    <mxCell id="1" parent="0"/>
    <!-- Diagram cells go here with parent="1" -->
  </root>
</mxGraphModel>
```

- Cell `id="0"` is the root layer
- Cell `id="1"` is the default parent layer
- All diagram elements use `parent="1"` unless using multiple layers

### Common styles

**Rounded rectangle:**
```xml
<mxCell id="2" value="Label" style="rounded=1;whiteSpace=wrap;" vertex="1" parent="1">
  <mxGeometry x="100" y="100" width="120" height="60" as="geometry"/>
</mxCell>
```

**Diamond (decision):**
```xml
<mxCell id="3" value="Condition?" style="rhombus;whiteSpace=wrap;" vertex="1" parent="1">
  <mxGeometry x="100" y="200" width="120" height="80" as="geometry"/>
</mxCell>
```

**Arrow (edge):**
```xml
<mxCell id="4" value="" style="edgeStyle=orthogonalEdgeStyle;" edge="1" source="2" target="3" parent="1">
  <mxGeometry relative="1" as="geometry"/>
</mxCell>
```

**Labeled arrow:**
```xml
<mxCell id="5" value="Yes" style="edgeStyle=orthogonalEdgeStyle;" edge="1" source="3" target="6" parent="1">
  <mxGeometry relative="1" as="geometry"/>
</mxCell>
```

### Useful style properties

| Property | Values | Use for |
|----------|--------|---------|
| `rounded=1` | 0 or 1 | Rounded corners |
| `whiteSpace=wrap` | wrap | Text wrapping |
| `fillColor=#dae8fc` | Hex color | Background color |
| `strokeColor=#6c8ebf` | Hex color | Border color |
| `fontColor=#333333` | Hex color | Text color |
| `fontSize=14` | Number | Font size in points |
| `shape=cylinder3` | shape name | Database cylinders |
| `shape=mxgraph.flowchart.document` | shape name | Document shapes |
| `ellipse` | style keyword | Circles/ovals |
| `rhombus` | style keyword | Diamonds |
| `edgeStyle=orthogonalEdgeStyle` | style keyword | Right-angle connectors |
| `edgeStyle=elbowEdgeStyle` | style keyword | Elbow connectors |
| `dashed=1` | 0 or 1 | Dashed lines |
| `swimlane` | style keyword | Swimlane containers |
| `container=1` | 0 or 1 | Group container |
| `collapsible=0` | 0 or 1 | Prevent collapse |

### Color palettes for consistent diagrams

**Blue theme** (default, professional):
- Fill: `#dae8fc`, Stroke: `#6c8ebf`

**Green theme** (success, output):
- Fill: `#d5e8d4`, Stroke: `#82b366`

**Orange theme** (warning, intermediate):
- Fill: `#fff2cc`, Stroke: `#d6b656`

**Red theme** (error, critical):
- Fill: `#f8cecc`, Stroke: `#b85450`

**Purple theme** (special, highlight):
- Fill: `#e1d5e7`, Stroke: `#9673a6`

**Gray theme** (neutral, background):
- Fill: `#f5f5f5`, Stroke: `#666666`

## CRITICAL: XML Well-formedness

- **NEVER use double hyphens (`--`) inside XML comments.** `--` is illegal inside `<!-- -->` per the XML spec and causes parse errors. Use single hyphens or rephrase.
- Escape special characters in attribute values: `&amp;`, `&lt;`, `&gt;`, `&quot;`
- Always use unique `id` values for each `mxCell`
- Keep `id` values as simple integers starting from 2 (0 and 1 are reserved)
