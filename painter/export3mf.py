"""3MF round-trip: per-triangle colors out, and back in to resume work.

A .3mf file is a zip containing an OPC package: [Content_Types].xml, _rels/.rels
and 3D/3dmodel.model (XML). Colors use the core-spec <basematerials> resource,
and every triangle references its palette entry via the p1 attribute — the
spec-compliant way to express per-face colors, understood by e.g. Bambu Studio
and Windows 3D Viewer.

Because the painting (the palette plus every face's color index) lives entirely
in that file, exporting IS saving the project: load_project() parses the same
XML back, preserving triangle order so the per-face colors line up exactly.
"""

from __future__ import annotations

import io
import time
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

from . import mesh as mesh_io
from .mesh import PaintMesh

CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
 <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>
"""

RELS = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>
"""


def _hex(color: np.ndarray) -> str:
    r, g, b = (np.clip(color, 0, 1) * 255).astype(int)
    return f"#{r:02X}{g:02X}{b:02X}"


def _parse_hex(s: str) -> tuple[float, float, float]:
    s = s.lstrip("#")
    return tuple(int(s[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def _local(tag: str) -> str:
    """Strip the XML namespace: '{...}triangle' -> 'triangle'."""
    return tag.rsplit("}", 1)[-1]


# --- export ------------------------------------------------------------------
def export(
    mesh: PaintMesh, face_colors: np.ndarray, palette: np.ndarray, path: str
) -> None:
    t0 = time.perf_counter()
    names = ["Unpainted"] + [f"Color {i}" for i in range(1, len(palette))]
    buf = io.StringIO()
    buf.write(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<model unit="millimeter" xml:lang="en-US" '
        'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">\n'
        " <resources>\n"
        '  <basematerials id="1">\n'
    )
    for name, color in zip(names, palette):
        buf.write(f'   <base name="{name}" displaycolor="{_hex(color)}" />\n')
    buf.write(
        "  </basematerials>\n"
        '  <object id="2" type="model" pid="1" pindex="0">\n'
        "   <mesh>\n    <vertices>\n"
    )
    for x, y, z in mesh.vertices:
        buf.write(f'     <vertex x="{x:.6g}" y="{y:.6g}" z="{z:.6g}" />\n')
    buf.write("    </vertices>\n    <triangles>\n")
    for (v1, v2, v3), c in zip(mesh.faces, face_colors):
        buf.write(
            f'     <triangle v1="{v1}" v2="{v2}" v3="{v3}" pid="1" p1="{c}" />\n'
        )
    buf.write(
        "    </triangles>\n   </mesh>\n  </object>\n"
        " </resources>\n"
        ' <build>\n  <item objectid="2" />\n </build>\n'
        "</model>\n"
    )

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", CONTENT_TYPES)
        z.writestr("_rels/.rels", RELS)
        z.writestr("3D/3dmodel.model", buf.getvalue())

    n_colors = len(np.unique(face_colors))
    print(
        f"Gemt {path}: {mesh.n_faces:,} faces, {n_colors} farver "
        f"på {time.perf_counter() - t0:.2f}s"
    )


# --- load --------------------------------------------------------------------
def load_project(path: str) -> tuple[PaintMesh, np.ndarray, np.ndarray | None]:
    """Parse a .3mf back into (mesh, per-face color index, palette colors).

    Focuses on the single-mesh core-spec layout this tool (and most simple 3MF
    files) use. Palette is None when the file carries no <basematerials>; in
    that case the colors are all 0 and only the geometry is restored. Raises on
    anything it can't make sense of so the caller can fall back to a plain
    geometry import.
    """
    t0 = time.perf_counter()
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        model = "3D/3dmodel.model"
        if model not in names:
            model = next((n for n in names if n.lower().endswith(".model")), None)
            if model is None:
                raise ValueError("ingen .model-fil i 3MF-pakken")
        root = ET.fromstring(z.read(model))

    resources = next((c for c in root if _local(c.tag) == "resources"), None)
    build = next((c for c in root if _local(c.tag) == "build"), None)
    if resources is None:
        raise ValueError("3MF mangler <resources>")

    materials: dict[str, list[tuple[float, float, float]]] = {}
    objects: dict[str, dict] = {}
    for res in resources:
        t = _local(res.tag)
        if t == "basematerials":
            cols = [
                _parse_hex(base.get("displaycolor"))
                for base in res
                if _local(base.tag) == "base" and base.get("displaycolor")
            ]
            materials[res.get("id")] = cols
        elif t == "object":
            mesh_el = next((c for c in res if _local(c.tag) == "mesh"), None)
            if mesh_el is None:
                continue  # components/transforms — ikke understøttet
            verts_el = next((c for c in mesh_el if _local(c.tag) == "vertices"), None)
            tris_el = next((c for c in mesh_el if _local(c.tag) == "triangles"), None)
            if verts_el is None or tris_el is None:
                continue
            verts = [
                (float(v.get("x")), float(v.get("y")), float(v.get("z")))
                for v in verts_el
                if _local(v.tag) == "vertex"
            ]
            faces, p1 = [], []
            for tri in tris_el:
                if _local(tri.tag) != "triangle":
                    continue
                faces.append(
                    (int(tri.get("v1")), int(tri.get("v2")), int(tri.get("v3")))
                )
                p = tri.get("p1")
                p1.append(int(p) if p is not None else -1)
            objects[res.get("id")] = {
                "verts": verts,
                "faces": faces,
                "p1": p1,
                "pid": res.get("pid"),
                "pindex": res.get("pindex"),
            }

    if not objects:
        raise ValueError("3MF har ingen mesh-objekter")

    # vælg build-item-objektet, ellers det første mesh
    oid = None
    if build is not None:
        item = next((c for c in build if _local(c.tag) == "item"), None)
        if item is not None:
            oid = item.get("objectid")
    if oid not in objects:
        oid = next(iter(objects))
    obj = objects[oid]

    mesh = mesh_io.build(obj["verts"], obj["faces"])

    # palette: objektets materiale, ellers det første der findes
    palette_cols: np.ndarray | None = None
    pid = obj["pid"]
    if pid in materials and materials[pid]:
        palette_cols = np.array(materials[pid], dtype="f4")
    elif materials:
        first = next((c for c in materials.values() if c), None)
        if first:
            palette_cols = np.array(first, dtype="f4")

    # per-face farveindeks: p1 pr. trekant, objektets pindex som fallback
    default = int(obj["pindex"]) if obj["pindex"] is not None else 0
    p1 = np.array([c if c >= 0 else default for c in obj["p1"]], dtype="i8")
    if palette_cols is not None and len(palette_cols):
        p1 = np.clip(p1, 0, len(palette_cols) - 1)
    else:
        p1 = np.zeros(len(obj["faces"]), dtype="i8")
    face_colors = p1.astype("u1")

    n_painted = int(np.count_nonzero(face_colors))
    print(
        f"Indlæst {path}: {mesh.n_faces:,} faces, {n_painted:,} malet "
        f"på {time.perf_counter() - t0:.2f}s"
    )
    return mesh, face_colors, palette_cols
