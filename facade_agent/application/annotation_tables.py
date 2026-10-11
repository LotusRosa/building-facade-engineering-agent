"""One project-specific 0/1 table format for browser labels and file imports."""
from __future__ import annotations

import csv
import hashlib
import io
import posixpath
import re
import zipfile
from collections import Counter
from typing import Any
from xml.etree import ElementTree as ET

from ..storage import canonical_json

NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
MAX_BYTES = 10 * 1024 * 1024
MAX_ROWS = 100_000


def class_headers(classes: list[dict[str, Any]]) -> list[str]:
    # Escape reserved column names without restricting the engineer's taxonomy.
    headers = [str(c["display_name"]).strip() for c in classes]
    if any(h.casefold() in {"filename", "no_defect"} for h in headers):
        headers = [f"class:{h}" for h in headers]
    if len({h.casefold() for h in headers}) != len(headers):
        raise ValueError("Class names must be distinct ignoring capitalization.")
    return headers


def _xml(payload: bytes) -> ET.Element:
    if b"\x00" in payload or b"<!DOCTYPE" in payload.upper() or b"<!ENTITY" in payload.upper():
        raise ValueError("XML entities are not supported in annotation workbooks.")
    return ET.fromstring(payload)


def read_table(content: bytes, filename: str) -> list[list[str]]:
    if not content or len(content) > MAX_BYTES:
        raise ValueError("Annotation table must be nonempty and at most 10 MB.")
    if filename.lower().endswith(".csv"):
        try:
            rows = list(csv.reader(io.StringIO(content.decode("utf-8-sig"), newline="")))
        except (UnicodeError, csv.Error) as exc:
            raise ValueError("Use a UTF-8 CSV or an Excel .xlsx workbook.") from exc
    elif filename.lower().endswith(".xlsx"):
        try:
            rows = _read_xlsx(content)
        except (zipfile.BadZipFile, ET.ParseError, KeyError, IndexError, RuntimeError, NotImplementedError) as exc:
            raise ValueError("Invalid Excel workbook; download the project template again.") from exc
    else:
        raise ValueError("Use .xlsx or UTF-8 .csv; both use the same project template.")
    if len(rows) > MAX_ROWS + 1:
        raise ValueError("Annotation table exceeds 100,000 rows.")
    return rows


def _read_xlsx(content: bytes) -> list[list[str]]:
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        members = archive.infolist()
        if len(members) > 1000 or sum(m.file_size for m in members) > 64 * 1024 * 1024:
            raise ValueError("Excel workbook exceeds the supported size.")
        if len({m.filename for m in members}) != len(members):
            raise ValueError("Excel workbook has duplicate archive members.")
        workbook = _xml(archive.read("xl/workbook.xml"))
        sheets = list(workbook.findall(f"{{{NS}}}sheets/{{{NS}}}sheet"))
        if len(sheets) != 1:
            raise ValueError("Use exactly one worksheet; extra worksheets are not accepted.")
        sheet = sheets[0]
        relationships = _xml(archive.read("xl/_rels/workbook.xml.rels"))
        relation = next((r for r in relationships if r.get("Id") == sheet.get(f"{{{REL}}}id")), None)
        if relation is None or relation.get("TargetMode") == "External":
            raise ValueError("Workbook worksheet relationship is invalid.")
        target = str(relation.get("Target", ""))
        path = posixpath.normpath(target.lstrip("/") if target.startswith("/") else "xl/" + target)
        if not path.startswith("xl/") or "\\" in path:
            raise ValueError("Workbook worksheet path is invalid.")
        strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared = _xml(archive.read("xl/sharedStrings.xml"))
            strings = ["".join(n.text or "" for n in si.iter(f"{{{NS}}}t")) for si in shared]
        document = _xml(archive.read(path))
        if document.find(f"{{{NS}}}mergeCells") is not None:
            raise ValueError("Merged cells are not supported in annotation tables.")
        rows = []
        for row in document.findall(f"{{{NS}}}sheetData/{{{NS}}}row"):
            cells: dict[int, str] = {}
            for cell in row:
                if cell.tag != f"{{{NS}}}c":
                    continue
                reference = re.fullmatch(r"([A-Z]+)[1-9][0-9]*", cell.get("r", ""))
                if reference is None:
                    raise ValueError("Worksheet cell address is invalid.")
                index = 0
                for letter in reference.group(1):
                    index = index * 26 + ord(letter) - 64
                if index > 1024 or index - 1 in cells:
                    raise ValueError("Worksheet column count or duplicate cell is invalid.")
                if cell.find(f"{{{NS}}}f") is not None:
                    raise ValueError("Use literal 0/1 values; formulas are not accepted.")
                value = cell.findtext(f"{{{NS}}}v", "")
                if cell.get("t") == "s":
                    try:
                        string_index = int(value)
                        if string_index < 0:
                            raise ValueError("Negative shared-string reference.")
                        value = strings[string_index]
                    except (ValueError, IndexError) as exc:
                        raise ValueError("Invalid shared-string reference.") from exc
                elif cell.get("t") == "inlineStr":
                    value = "".join(n.text or "" for n in cell.iter(f"{{{NS}}}t"))
                elif cell.get("t") == "e":
                    raise ValueError("Fix Excel error cells before importing labels.")
                cells[index - 1] = value
            rows.append([cells.get(i, "") for i in range(max(cells, default=-1) + 1)])
            if len(rows) > MAX_ROWS + 1:
                raise ValueError("Annotation table exceeds 100,000 rows.")
        return rows


def write_table(rows: list[list[Any]], format: str) -> bytes:
    if format == "csv":
        if any(str(value).lstrip().startswith(("=", "+", "-", "@")) for row in rows for value in row):
            raise ValueError("Use Excel export for filenames or class names beginning with formula characters.")
        stream = io.StringIO(newline="")
        csv.writer(stream).writerows(rows)
        return stream.getvalue().encode("utf-8-sig")
    if format != "xlsx":
        raise ValueError("Table format must be xlsx or csv.")
    ET.register_namespace("", NS)
    sheet = ET.Element(f"{{{NS}}}worksheet")
    data = ET.SubElement(sheet, f"{{{NS}}}sheetData")
    for row_index, values in enumerate(rows, 1):
        row = ET.SubElement(data, f"{{{NS}}}row", r=str(row_index))
        for col_index, value in enumerate(values, 1):
            n, column = col_index, ""
            while n:
                n, remainder = divmod(n - 1, 26)
                column = chr(65 + remainder) + column
            cell = ET.SubElement(row, f"{{{NS}}}c", r=f"{column}{row_index}", t="inlineStr")
            text = ET.SubElement(ET.SubElement(cell, f"{{{NS}}}is"), f"{{{NS}}}t")
            text.text = str(value)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>')
        archive.writestr("_rels/.rels", '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        archive.writestr("xl/workbook.xml", f'<workbook xmlns="{NS}" xmlns:r="{REL}"><sheets><sheet name="labels" sheetId="1" r:id="rId1"/></sheets></workbook>')
        archive.writestr("xl/_rels/workbook.xml.rels", f'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="{REL}/worksheet" Target="worksheets/sheet1.xml"/></Relationships>')
        archive.writestr("xl/worksheets/sheet1.xml", ET.tostring(sheet, encoding="utf-8", xml_declaration=True))
    return buffer.getvalue()


def parse_labels(content: bytes, filename: str, classes: list[dict], images: list[dict]) -> dict:
    rows = read_table(content, filename)
    if not rows:
        raise ValueError("Annotation table is empty.")
    expected = ["filename", *class_headers(classes), "no_defect"]
    headers = [h.strip().casefold() for h in rows[0]]
    if len(set(headers)) != len(headers) or set(headers) != {h.casefold() for h in expected}:
        raise ValueError("Table columns must match the project template: " + ", ".join(expected))
    positions = [headers.index(h.casefold()) for h in expected]
    counts = Counter(str(i["filename"]).casefold() for i in images)
    if any(n > 1 for n in counts.values()):
        raise ValueError("Image filenames must be unique across the selected folder and subfolders.")
    by_name = {str(i["filename"]).casefold(): i for i in images}
    entries, seen = [], set()
    for number, row in enumerate(rows[1:], 2):
        if not any(str(v).strip() for v in row):
            continue
        if len(row) > len(headers) and any(v.strip() for v in row[len(headers):]):
            raise ValueError(f"Row {number}: unexpected extra columns.")
        row = row + [""] * max(0, len(headers) - len(row))
        name, *values = [row[i].strip() for i in positions]
        key = name.casefold()
        if key not in by_name or key in seen:
            raise ValueError(f"Row {number}: missing image or repeated filename: {name}")
        seen.add(key)
        if all(v == "" for v in values):
            continue
        if any(v not in {"0", "1", "0.0", "1.0"} for v in values):
            raise ValueError(f"Row {number} ({name}): fill every label with 0/1, or leave all labels blank.")
        flags = [v in {"1", "1.0"} for v in values]
        if flags[-1] == any(flags[:-1]):
            raise ValueError(f"Row {number} ({name}): choose defects or no_defect=1, exclusively.")
        entries.append({"image_id": by_name[key]["image_id"], "class_ids": [c["class_id"] for c, flag in zip(classes, flags[:-1]) if flag], "no_defect": flags[-1]})
    complete = {i["image_id"] for i in images if i.get("annotation_status") == "complete"}
    updated = {e["image_id"] for e in entries}
    return {"entries": entries, "image_count": len(images), "imported_labels": len(entries), "overwrite_count": len(complete & updated), "remaining_unlabeled": len(images) - len(complete | updated)}


class AnnotationTableService:
    def __init__(self, store: Any) -> None:
        self.store = store

    def export(self, project_id: str, dataset_id: str | None, format: str, *, template: bool = False) -> bytes:
        project = self.store.get_project(project_id)
        classes = project["classes"]
        rows: list[list[Any]] = [["filename", *class_headers(classes), "no_defect"]]
        if dataset_id:
            dataset = self.store.get_dataset(dataset_id)
            if dataset["project_id"] != project_id:
                raise PermissionError("Dataset belongs to another project.")
            for image in self.store.list_images(dataset_id):
                labeled = not template and image["annotation_status"] == "complete"
                values = [int(c["class_id"] in image["class_ids"]) for c in classes] + [int(image["no_defect"])] if labeled else [""] * (len(classes) + 1)
                rows.append([image["filename"], *values])
        return write_table(rows, format)

    def import_table(self, dataset_id: str, content: bytes, filename: str, *, confirmed: bool = False, preview_token: str = "") -> dict:
        dataset = self.store.get_dataset(dataset_id)
        if dataset["status"] != "open":
            raise PermissionError("Frozen or validated labels are read-only.")
        classes = self.store.get_project(dataset["project_id"])["classes"]
        images = self.store.list_images(dataset_id)
        preview = parse_labels(content, filename, classes, images)
        revision = self.store.annotation_revision(images)
        token = hashlib.sha256(content + canonical_json([dataset_id, revision, preview]).encode()).hexdigest()
        if confirmed:
            if preview_token != token:
                raise PermissionError("Labels or images changed. Preview the table again before confirming.")
            self.store.save_annotations(dataset_id=dataset_id, entries=preview["entries"], actor_id="browser_engineer", expected_revision=revision)
        result = {key: value for key, value in preview.items() if key != "entries"} | {"applied": confirmed, "preview_token": token}
        if confirmed and not preview["remaining_unlabeled"] and preview["image_count"]:
            result["validation"] = self.store.validate_dataset(dataset_id=dataset_id)
        return result
