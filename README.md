# 3mf Painter

Farvelæg 3D-modeller til fler-farvet 3D-print. GPU-accelereret (moderngl/OpenGL til rendering, PyTorch CUDA til mesh-analyse).

## Kørsel

```powershell
.\.venv\Scripts\python.exe -m painter [model.stl]
```

Uden argument vises en demo-torus.

## Kontroller

| Input | Handling |
|---|---|
| File → Import STL... | Indlæs model (STL/3MF/OBJ) |
| File → Export 3MF... | Gem farvelagt model som .3mf |
| Ctrl+Z / Ctrl+Y | Undo / redo (pr. strøg) |
| Venstre træk | **Mal** med aktuel farve (pensel) eller **fyld region** (fill mode) |
| F | Skift mellem pensel og fill mode |
| M | Mirror mode: maling spejles over modellens symmetriplan |
| Højre træk | Rotér |
| Shift+træk / midterknap | Panorér |
| Scroll | Zoom |
| Ctrl+scroll | Penselstørrelse |
| 1–8 | Vælg farve |
| 0 | Viskelæder (umalet) |
| C | Ryd al maling |
| Esc | Luk |

## Roadmap

1. ✅ GPU-viewer (STL/3MF/OBJ-indlæsning, orbit-kamera, flat shading)
2. ✅ GPU-picking via face-ID-buffer + paint-brush
3. ✅ Auto-segmentering: kantdetektion via dihedral-vinkler (PyTorch, CUDA/CPU)
4. ✅ Symmetri-detektion (PCA-kandidatplaner + KD-træ-verifikation) og spejlet maling
5. ✅ 3MF-eksport med farvedata (core-spec basematerials, per-triangle)

## Arkitektur

- `painter/mesh.py` — indlæsning; mesh "unweldes" så hver face har egen normal, farve og face-ID (kræves til GPU-picking)
- `painter/camera.py` — orbit-kamera
- `painter/viewer.py` — moderngl/glfw-vindue og shaders
- `painter/painting.py` — farvepalette, per-face farvetilstand (GPU-tekstur) og offscreen face-ID-picking
- `painter/segmentation.py` — region-segmentering: dihedral-vinkler + connected components (torch, device-agnostisk)
- `painter/compute.py` — runtime GPU-detektion (CUDA hvis muligt, ellers CPU) og tuning
- `painter/export3mf.py` — 3MF-skrivning (OPC-zip med XML, per-triangle farver via basematerials)
- `painter/symmetry.py` — spejlplan-detektion: PCA-kandidater scoret med KD-træ-nearest-neighbor og normal-konsistens
