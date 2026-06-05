# Brain MRI Segmentation Web UI

React/Vite interface for the brain MRI tumor segmentation demo.

## Run

Start the API from the project root:

```powershell
.venv\Scripts\python.exe -m brain_mri_segmentation.api.main --config configs/inference.toml
```

Start the frontend:

```powershell
cd web
npm install
npm run dev
```

Open `http://127.0.0.1:5173`.

The UI expects the API at `http://127.0.0.1:8000` by default. You can override it with `VITE_API_URL`.
