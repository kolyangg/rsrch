#!/usr/bin/env node
/* Build the four-page CL39 / CL39-R2 architecture deck from one shared SVG layout. */

const fs = require("fs");
const path = require("path");
const BUNDLED_MODULES = process.platform === "win32"
  ? "C:\\Users\\ogure\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\node\\node_modules"
  : "/mnt/c/Users/ogure/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules";
const pptxgen = require(path.join(BUNDLED_MODULES, "pptxgenjs"));
const sharp = require(path.join(BUNDLED_MODULES, "sharp"));
const { PDFDocument } = require(path.join(BUNDLED_MODULES, "pdf-lib"));

const ROOT = path.resolve(__dirname, "../..");
const OUT_PPTX = path.join(ROOT, "analysis/2026-08-26_CL39_R2_architecture_schemes.pptx");
const OUT_PDF = path.join(ROOT, "analysis/assets/2026-08-26_CL39_R2_architecture_schemes.pdf");
const PREVIEW_DIR = path.join(ROOT, "analysis/assets/cl39_r2_architecture_schemes_20260826");

const C = {
  bg: "#F1F3F4",
  teal: "#006982",
  tealDark: "#00536A",
  green: "#008A5B",
  greenDark: "#006F49",
  paleGreen: "#E8F3EE",
  orange: "#D87900",
  orangeDark: "#A95D00",
  paleOrange: "#FFE4B0",
  ink: "#24323A",
  muted: "#53636C",
  line: "#087E59",
  white: "#FFFFFF",
  card: "#F8F9FA",
  cardStroke: "#BBC3C8",
  change: "#7C3AED",
  paleChange: "#F1EAFE",
};

function esc(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function lineText(x, y, lines, opts = {}) {
  const {
    size = 22,
    fill = C.ink,
    weight = 500,
    anchor = "middle",
    lineGap = Math.round(size * 1.22),
    italic = false,
    family = "Arial, Helvetica, sans-serif",
  } = opts;
  const start = y - ((lines.length - 1) * lineGap) / 2;
  const spans = lines
    .map((line, index) => `<tspan x="${x}" y="${start + index * lineGap}">${esc(line)}</tspan>`)
    .join("");
  return `<text text-anchor="${anchor}" font-family="${family}" font-size="${size}" font-weight="${weight}" font-style="${italic ? "italic" : "normal"}" fill="${fill}">${spans}</text>`;
}

function roundedBox(x, y, w, h, lines, opts = {}) {
  const {
    fill = C.green,
    stroke = C.greenDark,
    strokeWidth = 2.5,
    radius = 18,
    textFill = C.white,
    textSize = 24,
    textWeight = 600,
    dash = "",
    shadow = true,
  } = opts;
  return [
    `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${radius}" fill="${fill}" stroke="${stroke}" stroke-width="${strokeWidth}"${dash ? ` stroke-dasharray="${dash}"` : ""}${shadow ? ' filter="url(#shadow)"' : ""}/>` ,
    lineText(x + w / 2, y + h / 2 + textSize * 0.18, lines, {
      size: textSize,
      fill: textFill,
      weight: textWeight,
      lineGap: Math.round(textSize * 1.18),
    }),
  ].join("");
}

function badge(x, y, letter, fill = C.teal) {
  return `<circle cx="${x}" cy="${y}" r="21" fill="${fill}" stroke="${C.white}" stroke-width="3" filter="url(#shadow)"/>${lineText(x, y + 7, [letter], { size: 20, fill: C.white, weight: 700 })}`;
}

function arrow(points, opts = {}) {
  const { color = C.line, width = 4, dash = "", marker = true } = opts;
  const pts = points.map(([x, y]) => `${x},${y}`).join(" ");
  return `<polyline points="${pts}" fill="none" stroke="${color}" stroke-width="${width}" stroke-linecap="round" stroke-linejoin="round"${dash ? ` stroke-dasharray="${dash}"` : ""}${marker ? ` marker-end="url(#arrow-${color.slice(1)})"` : ""}/>`;
}

function noteCard(x, y, w, h, letter, lines, orange = false) {
  const accent = orange ? C.orange : C.teal;
  return [
    `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="12" fill="${C.card}" stroke="${C.cardStroke}" stroke-width="2" filter="url(#smallShadow)"/>`,
    badge(x + 20, y + 22, letter, accent),
    lineText(x + w / 2, y + h / 2 + 20, lines, {
      anchor: "middle",
      size: 13,
      weight: 500,
      lineGap: 16,
      fill: C.ink,
    }),
  ].join("");
}

const slides = [
  {
    code: "CL39",
    title: "HardcaseBranchedAttnProcessor (CL39) - Entropy-Routed Frequency Correction",
    subtitle: "Same doubled batch and branch Q/K/V • target Q used in both messages • both messages receive Wₒ before routing",
    routeTitle: "ROUTE FINISHED MESSAGES",
    routeLines: [
      "CL39",
      "Y = N + S·C₀·(gᴸ(p)L + gᴴ(p)H)",
      "D = R − N ;  D = L + H  (Gaussian split)",
      "C₀ = clip[1 − .75σ((E−.75)/.08), .25, 1]",
    ],
    changeTitle: "KEY DIFFERENCE",
    changeLines: [
      "Native N remains explicit; only the reference-derived correction",
      "is frequency-shaped and entropy-confidence-routed.",
    ],
    changeKind: "base",
    fNote: ["CL39 routes", "C₀·(gᴸL+gᴴH); N stays", "the denoising anchor."],
  },
  {
    code: "CL39-R2-A",
    title: "HardcaseBranchedAttnProcessor (CL39-R2-A) - Training-Only Reference Ownership",
    subtitle: "CL39 is unchanged at validation/inference • only a coherent, deterministic training-time ownership obligation is added",
    routeTitle: "ROUTE FINISHED MESSAGES",
    routeLines: [
      "YCL39 = N + S·C₀·(gᴸL + gᴴH)",
      "D = R − N = L + H ; C₀ unchanged",
      "α = Iselected·clip((step−2000)/4000, 0, 1)",
      "Ytrain = (1−α)YCL39 + α[N + S(R−N)]",
      "Yeval = YCL39  (α = 0)",
    ],
    changeTitle: "R2-A — ONLY CHANGE",
    changeLines: [
      "12.5% stateless selected steps",
      "one coherent α across all 36 up0/up1 processors",
      "diffusion loss occasionally depends on raw R",
      "no new parameters; validation/inference α=0",
    ],
    changeKind: "A",
    fNote: ["R2-A moves selected", "training steps toward raw R;", "inference remains CL39."],
  },
  {
    code: "CL39-R2-B",
    title: "HardcaseBranchedAttnProcessor (CL39-R2-B) - Bounded Low/High Reliability",
    subtitle: "CL39 entropy confidence stays as the base gate • one zero-initialized detached-feature MLP adds bounded band-specific corrections",
    routeTitle: "ROUTE FINISHED MESSAGES",
    routeLines: [
      "D = R − N = L + H ; C₀ unchanged",
      "zᴸ,zᴴ = MLP₆→₁₆→₂(stopgrad x)",
      "Cᴸ/ᴴ = clip[C₀ + .20·tanh(zᴸ/ᴴ), .25, 1]",
      "Y = N + S·(CᴸgᴸL + CᴴgᴴH)",
      "x: valid mass, conditional entropy, cos(N,R),",
      "log RMS(L/N), log RMS(H/N), progress",
    ],
    changeTitle: "R2-B — ONLY CHANGE",
    changeLines: [
      "Per-processor 6→16→2 reliability MLP; all six inputs detached",
      "zero final layer ⇒ Cᴸ=Cᴴ=C₀ at initialization",
      "+146 parameters × 36 selected processors",
    ],
    changeKind: "B",
    fNote: ["R2-B learns separate", "bounded Cᴸ/Cᴴ from", "detached features."],
  },
  {
    code: "CL39-R2-C",
    title: "HardcaseBranchedAttnProcessor (CL39-R2-C) - Fixed Low/High Face-RMS Caps",
    subtitle: "CL39 routing and entropy confidence are unchanged • only rare low/high residual tails are capped before gains and confidence",
    routeTitle: "ROUTE FINISHED MESSAGES",
    routeLines: [
      "D = R − N = L + H ; C₀ unchanged",
      "sᴸ=min[1, .90·rmsface(N)/(rmsface(L)+ε)]",
      "sᴴ=min[1, .45·rmsface(N)/(rmsface(H)+ε)]",
      "L̂=sᴸL ; Ĥ=sᴴH",
      "Y = N + S·C₀·(gᴸL̂ + gᴴĤ)",
    ],
    changeTitle: "R2-C — ONLY CHANGE",
    changeLines: [
      "Per-sample detached face-RMS caps",
      "kᴸ=0.90 and kᴴ=0.45",
      "act on raw L/H before existing gains and confidence",
      "no new parameters",
    ],
    changeKind: "C",
    fNote: ["R2-C caps low/high", "face-RMS tails before", "existing gains and C₀."],
  },
];

function buildSvg(spec, pageNumber) {
  const markers = [C.line, C.teal, C.orange, C.change].map((color) => {
    const id = color.slice(1);
    return `<marker id="arrow-${id}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="${color}"/></marker>`;
  }).join("");

  const changeIsBase = spec.changeKind === "base";
  const changeFill = changeIsBase ? C.paleGreen : C.paleChange;
  const changeStroke = changeIsBase ? C.orange : C.change;
  const changeDash = changeIsBase ? "" : "10 7";

  const notes = [
    ["A", ["Same doubled", "target/reference", "hidden-token inputs."], false],
    ["B", ["One full target Q", "is used in both", "target messages."], true],
    ["C", ["Native N receives Wₒ", "before any", "routing."], false],
    ["D", ["Reference face supplies", "explicit K/V; pose", "adapt is zero."], true],
    ["E", ["Soft target-mask S;", "transition rings avoid", "a hard boundary."], true],
    ["F", spec.fNote, true],
    ["G", ["Reference row keeps", "full self-attention", "and its own Wₒ."], false],
    ["H", ["Concat, dropout,", "residual, rescale;", "no second Wₒ."], true],
  ];

  const body = [];
  // Connectors first, so every box remains visually on top.
  body.push(arrow([[260, 285], [320, 250]], { color: C.line }));
  body.push(arrow([[260, 315], [320, 365]], { color: C.line }));
  body.push(arrow([[500, 250], [565, 270]], { color: C.line }));
  body.push(arrow([[500, 365], [565, 315]], { color: C.line }));
  body.push(arrow([[260, 565], [320, 530]], { color: C.line }));
  body.push(arrow([[525, 530], [565, 570]], { color: C.line }));
  body.push(arrow([[525, 640], [565, 610]], { color: C.line }));
  body.push(arrow([[500, 250], [535, 250], [535, 550], [565, 550]], { color: C.line }));
  body.push(arrow([[260, 610], [300, 610], [300, 760], [320, 760]], { color: C.line }));
  body.push(arrow([[525, 760], [565, 760]], { color: C.line }));
  body.push(arrow([[820, 290], [1080, 290], [1080, 330], [1115, 330]], { color: C.line }));
  body.push(arrow([[820, 590], [1000, 590], [1000, 515], [1115, 515]], { color: C.line }));
  body.push(arrow([[1075, 260], [1115, 260]], { color: C.orange }));
  body.push(arrow([[1595, 420], [1645, 420]], { color: C.line }));
  body.push(arrow([[820, 760], [1605, 760], [1605, 560], [1645, 560]], { color: C.line }));
  body.push(arrow([[1355, 620], [1355, 655]], { color: changeStroke, dash: changeDash || undefined }));
  if (!changeIsBase) {
    if (spec.changeKind === "A") {
      body.push(arrow([[790, 620], [860, 705]], { color: C.change, dash: "10 7" }));
      body.push(arrow([[1110, 705], [1110, 560], [1135, 560]], { color: C.change, dash: "10 7" }));
    } else if (spec.changeKind === "B") {
      body.push(arrow([[790, 330], [885, 655]], { color: C.change, dash: "10 7" }));
      body.push(arrow([[790, 590], [980, 655]], { color: C.change, dash: "10 7" }));
      body.push(arrow([[1370, 655], [1370, 600]], { color: C.change, dash: "10 7" }));
    } else if (spec.changeKind === "C") {
      body.push(arrow([[1000, 515], [1000, 655]], { color: C.change, dash: "10 7" }));
      body.push(arrow([[1370, 655], [1370, 600]], { color: C.change, dash: "10 7" }));
    }
  }

  // Labels and message blocks.
  body.push(lineText(80, 195, ["TARGET HALF"], { anchor: "start", size: 17, fill: C.greenDark, weight: 700 }));
  body.push(lineText(80, 485, ["REFERENCE HALF"], { anchor: "start", size: 17, fill: C.greenDark, weight: 700 }));
  body.push(roundedBox(55, 220, 205, 150, ["target hidden", "T"], { textSize: 28 }));
  body.push(roundedBox(320, 210, 180, 80, ["FULL Qₙ(T)", "used in both"], { fill: C.paleGreen, stroke: C.orange, textFill: C.ink, textSize: 21, textWeight: 700 }));
  body.push(roundedBox(320, 320, 180, 100, ["Kₙ(T)", "Vₙ(T)"], { textSize: 25 }));
  body.push(roundedBox(565, 235, 255, 125, ["N = Wₒ SDPA", "(Qₙ(T), Kₙ(T), Vₙ(T))"], { textSize: 22 }));
  body.push(roundedBox(55, 500, 205, 150, ["reference hidden", "Hᵣ"], { textSize: 28 }));
  body.push(roundedBox(320, 485, 205, 90, ["Hᵣ^face =", "Hᵣ ⊙ Mᵣ"], { fill: C.paleGreen, stroke: C.greenDark, textFill: C.ink, textSize: 22 }));
  body.push(roundedBox(320, 595, 205, 95, ["Kᵣ(Hᵣ^face)", "Vᵣ(Hᵣ^face)"], { textSize: 22 }));
  body.push(roundedBox(565, 525, 255, 130, ["R = Wₒ SDPA", "(Qₙ(T), Kᵣ, Vᵣ)"], { textSize: 23 }));
  body.push(roundedBox(320, 720, 205, 90, ["Qᵣ(Hᵣ)", "Kᵣ(Hᵣ), Vᵣ(Hᵣ)"], { textSize: 20 }));
  body.push(roundedBox(565, 710, 255, 105, ["Hᵣ^next = Wₒ SDPA", "(Qᵣ, Kᵣ, Vᵣ)"], { textSize: 20 }));
  body.push(roundedBox(875, 210, 200, 100, ["soft router S", "from target mask M"], { fill: C.paleGreen, stroke: C.orange, textFill: C.ink, textSize: 22 }));

  // Route block.
  body.push(`<rect x="1115" y="190" width="480" height="58" rx="15" fill="${C.orange}" filter="url(#shadow)"/>`);
  body.push(lineText(1138, 226, [spec.routeTitle], { anchor: "start", size: 23, fill: C.white, weight: 700 }));
  body.push(roundedBox(1115, 235, 480, 375, spec.routeLines, {
    fill: C.paleOrange,
    stroke: C.orange,
    textFill: C.ink,
    textSize: spec.code === "CL39-R2-B" ? 18 : (spec.code === "CL39" ? 22 : 20),
    textWeight: 550,
    radius: 28,
  }));

  body.push(roundedBox(1645, 300, 220, 300, ["concat", "[T_hard ; Hᵣ^next]", "", "output dropout only", "(routing already used Wₒ)", "", "reshape + residual", "+ rescale"], { textSize: 21 }));

  // Minimal-change callout: base key or the one R2 intervention.
  body.push(roundedBox(850, 655, 610, 140, [spec.changeTitle, ...spec.changeLines], {
    fill: changeFill,
    stroke: changeStroke,
    dash: changeDash,
    textFill: C.ink,
    textSize: changeIsBase ? 19 : 16,
    textWeight: 560,
    radius: 16,
  }));

  // A-H badges on the graph.
  body.push(badge(55, 220, "A", C.teal));
  body.push(badge(320, 210, "B", C.orange));
  body.push(badge(565, 235, "C", C.orange));
  body.push(badge(320, 485, "D", C.orange));
  body.push(badge(875, 210, "E", C.orange));
  body.push(badge(1115, 190, "F", C.orange));
  body.push(badge(320, 720, "G", C.teal));
  body.push(badge(1645, 300, "H", C.orange));

  // Bottom explanation cards.
  const cardY = 865;
  const cardW = 220;
  const gap = 15;
  const startX = 27;
  notes.forEach(([letter, lines, orange], index) => {
    body.push(noteCard(startX + index * (cardW + gap), cardY, cardW, 145, letter, lines, orange));
  });

  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="1920" height="1080" viewBox="0 0 1920 1080">
  <defs>
    ${markers}
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="150%"><feDropShadow dx="0" dy="5" stdDeviation="5" flood-color="#000000" flood-opacity="0.22"/></filter>
    <filter id="smallShadow" x="-20%" y="-20%" width="140%" height="150%"><feDropShadow dx="0" dy="3" stdDeviation="3" flood-color="#000000" flood-opacity="0.16"/></filter>
  </defs>
  <rect width="1920" height="1080" fill="${C.bg}"/>
  <rect x="25" y="20" width="1870" height="100" fill="${C.teal}" stroke="${C.tealDark}" stroke-width="3" filter="url(#shadow)"/>
  ${lineText(960, 86, [spec.title], { size: 36, fill: C.white, weight: 700 })}
  ${lineText(960, 153, [spec.subtitle], { size: 16, fill: C.ink, weight: 500, italic: true })}
  ${body.join("\n")}
  ${lineText(960, 1048, ["Source: clean_full • attn_processor_cleanest.py • CL39 / CL39-R2 configs • 26 Aug 2026"], { size: 13, fill: C.muted, weight: 500, italic: true })}
  ${lineText(1875, 1048, [String(pageNumber)], { size: 18, fill: C.muted, weight: 600 })}
</svg>`;
}

async function main() {
  fs.mkdirSync(path.dirname(OUT_PPTX), { recursive: true });
  fs.mkdirSync(path.dirname(OUT_PDF), { recursive: true });
  fs.mkdirSync(PREVIEW_DIR, { recursive: true });

  const pptx = new pptxgen();
  pptx.layout = "LAYOUT_WIDE";
  pptx.author = "Codex / PhotoMaker branched-attention research";
  pptx.subject = "CL39 and three independent CL39-R2 architecture schemes";
  pptx.title = "CL39 / CL39-R2 architecture schemes";
  pptx.company = "PhotoMaker BA research";
  pptx.lang = "en-US";
  pptx.theme = {
    headFontFace: "Arial",
    bodyFontFace: "Arial",
    lang: "en-US",
  };

  const pngs = [];
  for (let i = 0; i < slides.length; i += 1) {
    const svg = buildSvg(slides[i], i + 1);
    const svgPath = path.join(PREVIEW_DIR, `slide_${i + 1}_${slides[i].code.replaceAll("-", "_")}.svg`);
    const pngPath = path.join(PREVIEW_DIR, `slide_${i + 1}_${slides[i].code.replaceAll("-", "_")}.png`);
    fs.writeFileSync(svgPath, svg, "utf8");
    const png = await sharp(Buffer.from(svg), { density: 180 })
      .resize(2560, 1440, { fit: "fill" })
      .png({ compressionLevel: 9 })
      .toBuffer();
    fs.writeFileSync(pngPath, png);
    pngs.push(png);

    const slide = pptx.addSlide();
    slide.background = { color: C.bg.slice(1) };
    slide.addImage({ data: `data:image/svg+xml;base64,${Buffer.from(svg).toString("base64")}`, x: 0, y: 0, w: 13.333, h: 7.5 });
    slide.addNotes(`Slide ${i + 1}: ${slides[i].code}. Derived from the CL39 chart format; only the declared R2 mechanism changes.`);
  }
  await pptx.writeFile({ fileName: OUT_PPTX });

  const pdf = await PDFDocument.create();
  pdf.setTitle("CL39 / CL39-R2 architecture schemes");
  pdf.setAuthor("Codex / PhotoMaker branched-attention research");
  pdf.setSubject("Four matched architecture diagrams: CL39, R2-A, R2-B, R2-C");
  for (const png of pngs) {
    const page = pdf.addPage([960, 540]);
    const image = await pdf.embedPng(png);
    page.drawImage(image, { x: 0, y: 0, width: 960, height: 540 });
  }
  fs.writeFileSync(OUT_PDF, await pdf.save({ useObjectStreams: true }));

  console.log(JSON.stringify({ pptx: OUT_PPTX, pdf: OUT_PDF, previews: PREVIEW_DIR }, null, 2));
}

main().catch((error) => {
  console.error(error.stack || String(error));
  process.exit(1);
});
