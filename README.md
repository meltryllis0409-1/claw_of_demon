# Claw of Demon (Map v6)

## Install (Windows PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
py -m game.ui.pygame_app
```

## Run
```bash
python -m game.ui.pygame_app
```

## Controls
- Map: click nodes to advance (current node first, then next nodes)
- Combat: click cards to select; Play / Discard
- Combat: Sort Value / Sort Suit
- Combat: Sigils panel is collapsible (Sigils << / >>)
- F11: toggle fullscreen
