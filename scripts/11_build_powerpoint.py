from __future__ import annotations

import csv
import html
import shutil
import zipfile
from pathlib import Path


EMU_PER_INCH = 914400
SLIDE_W = 13.333
SLIDE_H = 7.5


def main() -> None:
    output = Path("presentacion_nlp_finanzas.pptx")
    reports = Path("runs/model-comparison/reports")
    temp = Path("runs/model-comparison/pptx_build")
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)

    media_dir = temp / "ppt" / "media"
    media_dir.mkdir(parents=True)
    slides_dir = temp / "ppt" / "slides"
    rels_dir = slides_dir / "_rels"
    rels_dir.mkdir(parents=True)

    images = copy_images(reports, media_dir)
    table_rows = read_compact_table(reports / "slides_model_table_compact.csv")

    slides = build_slides(table_rows, images)
    write_base_package(temp, len(slides))

    for index, slide in enumerate(slides, start=1):
        write_slide(slides_dir / f"slide{index}.xml", slide)
        write_slide_rels(rels_dir / f"slide{index}.xml.rels", slide)

    zip_pptx(temp, output)
    shutil.rmtree(temp)
    print(f"Saved PowerPoint to {output}")


def copy_images(reports: Path, media_dir: Path) -> dict[str, str]:
    image_specs = {
        "model_comparison": reports / "model_comparison.png",
        "training_loss": reports / "training_loss_curves.png",
        "eval_loss": reports / "eval_loss_curves.png",
        "semantic_baseline": reports / "selected_confusion_matrices" / "semantic_baseline_gpt2_no_finetune.png",
        "semantic_finetuned": reports / "selected_confusion_matrices" / "semantic_finetuned_nvda_amd.png",
        "financial_direction": reports / "selected_confusion_matrices" / "financial_direction_features.png",
    }
    copied = {}
    for idx, (name, path) in enumerate(image_specs.items(), start=1):
        if not path.exists():
            continue
        target = media_dir / f"image{idx}.png"
        shutil.copyfile(path, target)
        copied[name] = target.name
    return copied


def read_compact_table(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.reader(file))


def build_slides(table_rows: list[list[str]], images: dict[str, str]) -> list[dict]:
    return [
        {
            "title": "Fine-tuning de GPT-2 para narrativas financieras",
            "subtitle": "Proyecto NLP aplicado: generacion, evaluacion semantica y senales exploratorias",
            "bullets": [
                "Modelo base: GPT-2",
                "Dominio: noticias financieras + datos de mercado",
                "Tesis: el fine-tuning mejora adaptacion al dominio, no demuestra prediccion robusta",
            ],
            "notes": "Abrir aclarando que no es un sistema de trading, sino un estudio de NLP aplicado.",
        },
        {
            "title": "Pregunta experimental",
            "bullets": [
                "Puede GPT-2 generar narrativas financieras plausibles?",
                "El fine-tuning mejora frente a GPT-2 base?",
                "El tono generado coincide con el de noticias reales?",
                "La senal generada se relaciona con la direccion real del activo?",
            ],
            "callout": "La clave: calidad textual, semantica y utilidad financiera no son lo mismo.",
        },
        {
            "title": "Conceptos de NLP usados",
            "bullets": [
                "Transfer learning: partir de GPT-2 preentrenado",
                "Language modeling causal: predecir el siguiente token",
                "Tokenizacion subword y embeddings contextuales",
                "Fine-tuning con prompts financieros",
                "Evaluacion generativa: ROUGE, BERTScore y FinBERT",
            ],
        },
        {
            "title": "Pipeline experimental",
            "bullets": [
                "1. Dataset FNSPID + datos de mercado",
                "2. Construccion de prompts con ventana de noticias previas",
                "3. Split temporal train / validation / test",
                "4. Fine-tuning de GPT-2",
                "5. Generacion sobre test",
                "6. Evaluacion textual, semantica y financiera",
            ],
            "callout": "El split temporal evita entrenar con informacion futura.",
        },
        {
            "title": "Formato prompt / target",
            "code": (
                "[TICKER: AMD]\n"
                "[DATE: 2023-04-18]\n"
                "[PRICE_CHANGE: -0.10%]\n"
                "[VOLUME_RATIO: 0.80]\n"
                "[RSI: 36.50]\n\n"
                "[PREVIOUS NEWS]\n"
                "- Interesting AMD Put And Call Options...\n"
                "- AMD Stock Sinks As Market Gains...\n\n"
                "[NEXT DAY NEWS]"
            ),
            "bullets": [
                "Tarea formulada como prompt/completion",
                "Algunos runs agregan sentimiento y direccion estructurada",
            ],
        },
        {
            "title": "Evaluacion multicapa",
            "bullets": [
                "Textual: ROUGE-L y BERTScore",
                "Semantica: FinBERT sobre noticia real vs generada",
                "Financiera: directional accuracy + coverage",
                "Estadistica: test binomial contra azar p = 0.5",
            ],
            "callout": "Directional accuracy sola puede enganar: siempre se lee con coverage y p-value.",
        },
        {
            "title": "Resultados principales",
            "table": table_rows,
            "callout": "El fine-tuning mejora calidad textual/semantica; la metrica financiera sigue siendo exploratoria.",
        },
        {
            "title": "Comparacion entre runs",
            "image": images.get("model_comparison"),
            "bullets": [
                "NVDA + AMD: mejor balance textual/semantico",
                "Direction features: mayor directional accuracy",
                "GPT-2 base: alerta sobre sesgos de la metrica financiera",
            ],
        },
        {
            "title": "Loss y perplexity",
            "images": [images.get("training_loss"), images.get("eval_loss")],
            "callout": "Menor loss/perplexity no garantiza mejor metrica downstream financiera.",
        },
        {
            "title": "Matrices semanticas",
            "images": [images.get("semantic_baseline"), images.get("semantic_finetuned")],
            "bullets": [
                "GPT-2 base: bajo match semantico",
                "Fine-tuned NVDA + AMD: mejora parcial",
                "Aun no supera consistentemente el baseline neutral",
            ],
        },
        {
            "title": "Metrica financiera",
            "image": images.get("financial_direction"),
            "bullets": [
                "Direction features: directional accuracy 0.556 con coverage 1.0",
                "p-value aproximado: 0.225",
                "Lectura correcta: senal exploratoria, no predictividad demostrada",
            ],
        },
        {
            "title": "Ejemplos cualitativos",
            "bullets": [
                "Fine-tuning mejora un caso de alineacion semantica: ROUGE-L 0.1786 vs 0.0305",
                "GPT-2 base puede acertar direccion con narrativa pobre",
                "Los ejemplos justifican la evaluacion multicapa",
            ],
            "callout": "Una narrativa puede sonar financiera sin ser semanticamente correcta.",
        },
        {
            "title": "Limitaciones",
            "bullets": [
                "Dataset de test chico para conclusiones financieras robustas",
                "Targets basados en titulares agregados, no articulos completos",
                "GPT-2 small tiende a generar texto generico/clickbait",
                "FinBERT es un evaluador automatico, no verdad absoluta",
                "No hay significancia estadistica fuerte en directional accuracy",
            ],
        },
        {
            "title": "Conclusiones",
            "bullets": [
                "GPT-2 fine-tuneado aprende vocabulario y formato financiero",
                "El fine-tuning mejora adaptacion textual/semantica frente a GPT-2 base",
                "Las metricas textuales, semanticas y financieras no son equivalentes",
                "La directional accuracy requiere coverage, p-value y baselines fuertes",
                "Aporte principal: pipeline reproducible y evaluacion critica multicapa",
            ],
        },
        {
            "title": "Proximos pasos",
            "bullets": [
                "Agregar baselines estadisticos financieros mas fuertes",
                "Probar modelos mas grandes o especializados en finanzas",
                "Mejorar control estructural de sentimiento/direccion",
                "Sumar evaluacion humana de plausibilidad",
                "Explorar atencion como interpretabilidad, con cautela",
            ],
            "callout": "Cierre: no vendemos prediccion; defendemos metodologia NLP aplicada.",
        },
    ]


def write_base_package(root: Path, slide_count: int) -> None:
    (root / "_rels").mkdir(exist_ok=True)
    (root / "docProps").mkdir(exist_ok=True)
    (root / "ppt" / "_rels").mkdir(parents=True, exist_ok=True)
    (root / "ppt" / "theme").mkdir(exist_ok=True)
    (root / "ppt" / "slideMasters" / "_rels").mkdir(parents=True, exist_ok=True)
    (root / "ppt" / "slideLayouts" / "_rels").mkdir(parents=True, exist_ok=True)

    write(root / "[Content_Types].xml", content_types(slide_count))
    write(root / "_rels" / ".rels", package_rels())
    write(root / "docProps" / "core.xml", core_props())
    write(root / "docProps" / "app.xml", app_props(slide_count))
    write(root / "ppt" / "presentation.xml", presentation_xml(slide_count))
    write(root / "ppt" / "_rels" / "presentation.xml.rels", presentation_rels(slide_count))
    write(root / "ppt" / "theme" / "theme1.xml", theme_xml())
    write(root / "ppt" / "slideMasters" / "slideMaster1.xml", slide_master_xml())
    write(root / "ppt" / "slideMasters" / "_rels" / "slideMaster1.xml.rels", slide_master_rels())
    write(root / "ppt" / "slideLayouts" / "slideLayout1.xml", slide_layout_xml())
    write(root / "ppt" / "slideLayouts" / "_rels" / "slideLayout1.xml.rels", slide_layout_rels())


def write_slide(path: Path, slide: dict) -> None:
    shapes = []
    shapes.append(textbox(0.55, 0.25, 12.2, 0.55, slide["title"], 30, bold=True, color="17324D"))
    y = 1.05
    if slide.get("subtitle"):
        shapes.append(textbox(0.65, y, 11.5, 0.45, slide["subtitle"], 17, color="4B5F73"))
        y += 0.55
    if slide.get("table"):
        shapes.append(table_shape(0.35, 1.1, 12.65, 3.95, slide["table"]))
        y = 5.25
    if slide.get("code"):
        shapes.append(textbox(0.75, 1.25, 6.15, 3.45, slide["code"], 14, font="Consolas", fill="EEF3F8"))
        y = 1.35
        if slide.get("bullets"):
            shapes.append(bullets(7.25, y, 5.2, 2.8, slide["bullets"], 18))
            y = 4.55
    elif slide.get("image"):
        shapes.append(picture(slide["image"], 0.65, 1.15, 7.7, 4.7))
        if slide.get("bullets"):
            shapes.append(bullets(8.7, 1.3, 3.9, 3.7, slide["bullets"], 17))
            y = 5.15
    elif slide.get("images"):
        imgs = [img for img in slide["images"] if img]
        if len(imgs) == 2:
            shapes.append(picture(imgs[0], 0.55, 1.15, 6.15, 4.5))
            shapes.append(picture(imgs[1], 6.85, 1.15, 6.15, 4.5))
            y = 5.75
        elif len(imgs) == 1:
            shapes.append(picture(imgs[0], 1.0, 1.1, 11.2, 4.7))
            y = 5.9
    elif slide.get("bullets") and not slide.get("code"):
        shapes.append(bullets(0.85, y, 11.7, 4.6, slide["bullets"], 22 if len(slide["bullets"]) <= 4 else 19))
        y = 5.6
    if slide.get("callout"):
        shapes.append(textbox(0.75, max(y, 5.65), 11.9, 0.85, slide["callout"], 20, bold=True, color="17324D", fill="DDEAF3"))

    write(path, slide_xml("\n".join(shapes)))


def write_slide_rels(path: Path, slide: dict) -> None:
    rels = [
        '<Relationship Id="rIdLayout" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>'
    ]
    image_names = []
    if slide.get("image"):
        image_names.append(slide["image"])
    for img in slide.get("images", []) or []:
        if img:
            image_names.append(img)
    for i, image_name in enumerate(image_names, start=1):
        rels.append(
            f'<Relationship Id="rIdImg{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/{image_name}"/>'
        )
    write(path, rels_xml(rels))


def textbox(x: float, y: float, w: float, h: float, text: str, size: int, bold: bool = False, color: str = "1F2933", font: str = "Aptos", fill: str | None = None) -> str:
    fill_xml = f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>' if fill else "<a:noFill/>"
    paragraphs = "".join(paragraph(line, size, bold, color, font) for line in str(text).splitlines())
    return f"""
<p:sp>
  <p:nvSpPr><p:cNvPr id="{new_id()}" name="TextBox"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
  <p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom>{fill_xml}</p:spPr>
  <p:txBody><a:bodyPr wrap="square" lIns="100000" tIns="60000" rIns="100000" bIns="60000"/><a:lstStyle/>{paragraphs}</p:txBody>
</p:sp>"""


def bullets(x: float, y: float, w: float, h: float, items: list[str], size: int) -> str:
    paras = "".join(bullet_paragraph(item, size) for item in items)
    return f"""
<p:sp>
  <p:nvSpPr><p:cNvPr id="{new_id()}" name="Bullets"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
  <p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/></p:spPr>
  <p:txBody><a:bodyPr wrap="square"/><a:lstStyle/>{paras}</p:txBody>
</p:sp>"""


def table_shape(x: float, y: float, w: float, h: float, rows: list[list[str]]) -> str:
    text = "\n".join(" | ".join(row[:8]) for row in rows[:6])
    return textbox(x, y, w, h, text, 10, font="Consolas", fill="F6F8FA")


def picture(image_name: str, x: float, y: float, w: float, h: float) -> str:
    rid = "rIdImg1"
    # If a slide has two images, PowerPoint resolves by order of pic elements and rel ids.
    rid = f"rIdImg{picture.counter}"
    picture.counter += 1
    return f"""
<p:pic>
  <p:nvPicPr><p:cNvPr id="{new_id()}" name="{escape(image_name)}"/><p:cNvPicPr/><p:nvPr/></p:nvPicPr>
  <p:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>
  <p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr>
</p:pic>"""


picture.counter = 1
_shape_id = 10


def new_id() -> int:
    global _shape_id
    _shape_id += 1
    return _shape_id


def paragraph(text: str, size: int, bold: bool, color: str, font: str) -> str:
    b = ' b="1"' if bold else ""
    return f'<a:p><a:r><a:rPr lang="es-AR" sz="{size * 100}"{b}><a:solidFill><a:srgbClr val="{color}"/></a:solidFill><a:latin typeface="{font}"/></a:rPr><a:t>{escape(text)}</a:t></a:r><a:endParaRPr lang="es-AR"/></a:p>'


def bullet_paragraph(text: str, size: int) -> str:
    return f'<a:p><a:pPr marL="342900" indent="-171450"><a:buChar char="•"/></a:pPr><a:r><a:rPr lang="es-AR" sz="{size * 100}"><a:solidFill><a:srgbClr val="1F2933"/></a:solidFill></a:rPr><a:t>{escape(text)}</a:t></a:r></a:p>'


def slide_xml(shapes: str) -> str:
    picture.counter = 1
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:spTree>
    <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
    <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
    {shapes}
  </p:spTree></p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>'''


def content_types(slide_count: int) -> str:
    slides = "\n".join(
        f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        for i in range(1, slide_count + 1)
    )
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Default Extension="png" ContentType="image/png"/>
<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>
<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>
{slides}
</Types>'''


def package_rels() -> str:
    return rels_xml([
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>',
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>',
        '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>',
    ])


def presentation_xml(slide_count: int) -> str:
    slide_ids = "\n".join(f'<p:sldId id="{255 + i}" r:id="rId{i}"/>' for i in range(1, slide_count + 1))
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rIdMaster"/></p:sldMasterIdLst>
<p:sldIdLst>{slide_ids}</p:sldIdLst>
<p:sldSz cx="{emu(SLIDE_W)}" cy="{emu(SLIDE_H)}" type="wide"/>
<p:notesSz cx="6858000" cy="9144000"/>
</p:presentation>'''


def presentation_rels(slide_count: int) -> str:
    rels = [
        '<Relationship Id="rIdMaster" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>',
        '<Relationship Id="rIdTheme" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/>',
    ]
    rels.extend(
        f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>'
        for i in range(1, slide_count + 1)
    )
    return rels_xml(rels)


def slide_master_xml() -> str:
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
<p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
<p:sldLayoutIdLst><p:sldLayoutId id="1" r:id="rId1"/></p:sldLayoutIdLst><p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles></p:sldMaster>'''


def slide_layout_xml() -> str:
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1"><p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld></p:sldLayout>'''


def slide_master_rels() -> str:
    return rels_xml([
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>',
        '<Relationship Id="rIdTheme" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>',
    ])


def slide_layout_rels() -> str:
    return rels_xml([
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>'
    ])


def theme_xml() -> str:
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="Simple"><a:themeElements><a:clrScheme name="Simple"><a:dk1><a:srgbClr val="111111"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1><a:dk2><a:srgbClr val="17324D"/></a:dk2><a:lt2><a:srgbClr val="EEF3F8"/></a:lt2><a:accent1><a:srgbClr val="2F6F7E"/></a:accent1><a:accent2><a:srgbClr val="C05746"/></a:accent2><a:accent3><a:srgbClr val="6B8E23"/></a:accent3><a:accent4><a:srgbClr val="8064A2"/></a:accent4><a:accent5><a:srgbClr val="4BACC6"/></a:accent5><a:accent6><a:srgbClr val="F79646"/></a:accent6><a:hlink><a:srgbClr val="0000FF"/></a:hlink><a:folHlink><a:srgbClr val="800080"/></a:folHlink></a:clrScheme><a:fontScheme name="Simple"><a:majorFont><a:latin typeface="Aptos Display"/></a:majorFont><a:minorFont><a:latin typeface="Aptos"/></a:minorFont></a:fontScheme><a:fmtScheme name="Simple"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="6350"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme></a:themeElements></a:theme>'''


def core_props() -> str:
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Presentacion NLP Finanzas</dc:title><dc:creator>Codex</dc:creator></cp:coreProperties>'''


def app_props(slide_count: int) -> str:
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"><Application>Microsoft PowerPoint</Application><Slides>{slide_count}</Slides></Properties>'''


def rels_xml(rels: list[str]) -> str:
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' + "".join(rels) + "</Relationships>"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def zip_pptx(root: Path, output: Path) -> None:
    if output.exists():
        output.unlink()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in root.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(root).as_posix())


def emu(inches: float) -> int:
    return int(inches * EMU_PER_INCH)


def escape(text: str) -> str:
    return html.escape(str(text), quote=False)


if __name__ == "__main__":
    main()
