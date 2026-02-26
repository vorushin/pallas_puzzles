---
name: drawio
description: Use when creating or editing draw.io diagrams — generates mxGraphModel XML, exports to JPG for visual review, iterates until elegant, then produces final SVG for the notebook
---

# Draw.io Diagram Skill

Generate draw.io diagrams as native `.drawio` files with visual review. Every diagram is exported to JPG and visually inspected, iterating until it is elegant, readable, and correct. The final deliverable is an SVG in the notebook.

**IMPORTANT: Always use JPG (not PNG) for the internal review loop.** Claude Code gets API Error 400 when reading PNG files exported by draw.io. JPG works reliably.

## Core Workflow

1. **Generate XML** in mxGraphModel format for the requested diagram
2. **Write the XML** to a `.drawio` file using the Write tool
3. **Export to JPG** using the draw.io CLI (`-f jpg`)
4. **Visually review** — use the Read tool on the JPG to inspect the rendered diagram
5. **Iterate** — if anything is off (overlaps, bad spacing, unclear flow, ugly styling), edit the `.drawio` file and re-export. Repeat until satisfied.
6. **Deliver** — once the diagram looks good:
   a. Export a **plain SVG** (no embedded diagram XML) to `images/<name>.drawio.svg`
   b. Move the `.drawio` source to `images/<name>.drawio` (keep as editable original)
   c. Add the SVG to the notebook as a markdown image using the raw GitHub URL:
      `![Alt text](https://raw.githubusercontent.com/vorushin/pallas_puzzles/master/images/<name>.drawio.svg)`
   d. Delete the temporary JPG (it was only for review)

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

## Final Output

Two files saved to `images/`:
- `images/<name>.drawio` — editable source (open in draw.io)
- `images/<name>.drawio.svg` — plain SVG for display (no embedded XML metadata)

The SVG is referenced in the notebook via raw GitHub URL for Colab rendering.

## draw.io CLI

### Locating the CLI

Try `drawio` first (works if on PATH), then fall back to:

- **macOS**: `/Applications/draw.io.app/Contents/MacOS/draw.io`

Use `which drawio` to check if it's on PATH before falling back.

### Export commands

For visual review (JPG):
```bash
drawio -x -f jpg -b 10 -o <name>.drawio.jpg <name>.drawio
```

For final deliverable (plain SVG, no embedded metadata):
```bash
drawio -x -f svg -b 10 -o images/<name>.drawio.svg <name>.drawio
```

Key flags:
- `-x` / `--export`: export mode
- `-f` / `--format`: output format (`jpg` for review, `svg` for final)
- `-o` / `--output`: output file path
- `-b` / `--border`: border width around diagram (default: 0)
- `-s` / `--scale`: scale the diagram size
- `--width` / `--height`: fit into specified dimensions (preserves aspect ratio)

Do NOT use `-e` / `--embed-diagram` for SVG — we want a clean SVG without embedded XML.

## File Naming

- Descriptive filename based on diagram content (e.g., `login-flow`, `database-schema`)
- Lowercase with hyphens for multi-word names
- Double extensions for exports: `name.drawio.svg`
- Both `.drawio` source and `.drawio.svg` final go in `images/`

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
