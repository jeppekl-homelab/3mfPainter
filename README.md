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
| File → Open... | Åbn model (STL/OBJ) eller malet projekt (.3mf med farver + palette) |
| File → Save project (3MF)... | Gem mesh + farver + palette som .3mf — kan genåbnes og redigeres |
| Ctrl+Z / Ctrl+Y | Undo / redo (pr. strøg) |
| Venstre træk | **Mal** med aktuel farve (pensel) eller **fyld region** (fill mode) |
| F | Skift mellem pensel og fill mode |
| M | Mirror mode: maling spejles over modellens symmetriplan |
| Højre træk | Rotér |
| Shift+træk / midterknap | Panorér |
| Scroll | Zoom |
| Ctrl+scroll | Penselstørrelse |
| 1–5 | Vælg male-farve (hver redigerbar via color picker i Tools-panelet) |
| 0 | Viskelæder (umalet) |
| C | Ryd al maling |
| Esc | Luk |

I fill mode fremhæves regionen under cursoren, så man kan se hvad et klik vil farve.

## Roadmap

1. ✅ GPU-viewer (STL/3MF/OBJ-indlæsning, orbit-kamera, flat shading)
2. ✅ GPU-picking via face-ID-buffer + paint-brush
3. ✅ Auto-segmentering: kantdetektion via dihedral-vinkler (PyTorch, CUDA/CPU)
4. ✅ Symmetri-detektion (PCA-kandidatplaner + KD-træ-verifikation) og spejlet maling
5. ✅ 3MF-eksport med farvedata (core-spec basematerials, per-triangle)
6. ✅ Region-hover-highlight i fill mode (forhåndsvisning af klik)
7. ✅ Gem/indlæs projekt i 3MF (round-trip af geometri, farver og palette)
8. ✅ Redigerbar palette: op til 5 male-farver via color picker

## Arkitektur

- `painter/mesh.py` — indlæsning; mesh "unweldes" så hver face har egen normal, farve og face-ID (kræves til GPU-picking)
- `painter/camera.py` — orbit-kamera
- `painter/viewer.py` — moderngl/glfw-vindue og shaders
- `painter/painting.py` — redigerbar palette, per-face farvetilstand (GPU-tekstur), fill-mode hover-highlight og offscreen face-ID-picking
- `painter/segmentation.py` — region-segmentering: dihedral-vinkler + connected components (torch, device-agnostisk)
- `painter/compute.py` — runtime GPU-detektion (CUDA hvis muligt, ellers CPU) og tuning
- `painter/export3mf.py` — 3MF round-trip: skrivning (OPC-zip med XML, per-triangle farver via basematerials) og indlæsning (`load_project` bevarer trekant-rækkefølge, gendanner farver + palette)
- `painter/symmetry.py` — spejlplan-detektion: PCA-kandidater scoret med KD-træ-nearest-neighbor og normal-konsistens
