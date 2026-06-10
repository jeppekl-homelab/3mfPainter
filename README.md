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
| Venstre træk | **Mal** med aktuel farve |
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
3. ⬜ Auto-segmentering: kantdetektion via dihedral-vinkler (PyTorch CUDA)
4. ⬜ Symmetri-detektion og farve-propagering
5. ⬜ 3MF-eksport med farvedata

## Arkitektur

- `painter/mesh.py` — indlæsning; mesh "unweldes" så hver face har egen normal, farve og face-ID (kræves til GPU-picking)
- `painter/camera.py` — orbit-kamera
- `painter/viewer.py` — moderngl/glfw-vindue og shaders
- `painter/painting.py` — farvepalette, per-face farvetilstand (GPU-tekstur) og offscreen face-ID-picking
