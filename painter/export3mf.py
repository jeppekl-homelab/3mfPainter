"""3MF export with per-triangle colors.

A .3mf file is a zip containing an OPC package: [Content_Types].xml, _rels/.rels
and 3D/3dmodel.model (XML). Colors use the core-spec <basematerials> resource,
and every triangle references its palette entry via the p1 attribute — the
spec-compliant way to express per-face colors, understood by e.g. Bambu Studio
and Windows 3D Viewer.
"""

from __future__ import annotations

import io
import time
import zipfile

import numpy as np

from .mesh import PaintMesh
from .painting import PALETTE

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

COLOR_NAMES = [
    "Unpainted", "Red", "Blue", "Green", "Yellow",
    "Orange", "Purple", "Black", "White",
]


def _hex(color: np.ndarray) -> str:
    r, g, b = (np.clip(color, 0, 1) * 255).astype(int)
    return f"#{r:02X}{g:02X}{b:02X}"


def export(mesh: PaintMesh, face_colors: np.ndarray, path: str) -> None:
    t0 = time.perf_counter()
    buf = io.StringIO()
    buf.write(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<model unit="millimeter" xml:lang="en-US" '
        'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">\n'
        " <resources>\n"
        '  <basematerials id="1">\n'
    )
    for name, color in zip(COLOR_NAMES, PALETTE):
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
        f"Eksporteret {path}: {mesh.n_faces:,} faces, {n_colors} farver "
        f"på {time.perf_counter() - t0:.2f}s"
    )
