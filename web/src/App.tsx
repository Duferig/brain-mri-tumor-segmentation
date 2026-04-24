import {
  Activity,
  Brain,
  CheckCircle2,
  Download,
  FileScan,
  FlaskConical,
  Play,
  RefreshCcw,
  Server,
  UploadCloud,
} from "lucide-react";
import { ChangeEvent, FormEvent, ReactNode, useEffect, useMemo, useState } from "react";
import {
  ExperimentSummary,
  ModelInfo,
  PredictionResponse,
  UploadFiles,
  fetchExperiments,
  fetchHealth,
  fetchModels,
  runPrediction,
} from "./api";

const DEFAULT_API = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";
const PLANE_LABELS: Record<string, string> = {
  axial: "Axial",
  coronal: "Coronal",
  sagittal: "Sagittal",
};

const MODALITIES: Array<{
  key: keyof UploadFiles;
  title: string;
  description: string;
  required: boolean;
}> = [
  { key: "t1", title: "T1", description: "анатомическая структура", required: true },
  { key: "t1ce", title: "T1ce", description: "контрастное усиление", required: true },
  { key: "t2", title: "T2", description: "жидкостные изменения", required: true },
  { key: "flair", title: "FLAIR", description: "отёк и патологические зоны", required: true },
  { key: "referenceSeg", title: "SEG", description: "эталонная маска для Dice", required: false },
];

function App() {
  const [apiBase, setApiBase] = useState(DEFAULT_API);
  const [apiStatus, setApiStatus] = useState<"checking" | "online" | "offline">("checking");
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [experiments, setExperiments] = useState<ExperimentSummary[]>([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [files, setFiles] = useState<UploadFiles>({
    t1: null,
    t1ce: null,
    t2: null,
    flair: null,
    referenceSeg: null,
  });
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [activePlane, setActivePlane] = useState("axial");
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const selectedModelInfo = models.find((model) => model.key === selectedModel);
  const selectedPreview = useMemo(() => {
    return prediction?.preview_images.find((image) => image.plane === activePlane)
      ?? prediction?.preview_images[0]
      ?? null;
  }, [activePlane, prediction]);
  const completedUploads = MODALITIES.filter((item) => item.required && files[item.key]).length;

  useEffect(() => {
    void refreshApiState();
  }, []);

  async function refreshApiState() {
    setApiStatus("checking");
    setError(null);
    try {
      const [health, modelList, experimentList] = await Promise.all([
        fetchHealth(apiBase),
        fetchModels(apiBase),
        fetchExperiments(apiBase),
      ]);
      setApiStatus(health.status === "ok" ? "online" : "offline");
      setModels(modelList);
      setExperiments(experimentList);
      setSelectedModel((current) => {
        const available = modelList.map((model) => model.key);
        return available.includes(current) ? current : health.available_models[0] || available[0] || "";
      });
    } catch (requestError) {
      setApiStatus("offline");
      setError(requestError instanceof Error ? requestError.message : "API недоступен");
    }
  }

  function handleFileChange(key: keyof UploadFiles, event: ChangeEvent<HTMLInputElement>) {
    const nextFile = event.target.files?.[0] ?? null;
    setFiles((current) => ({ ...current, [key]: nextFile }));
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setIsRunning(true);
    setError(null);
    try {
      const result = await runPrediction(apiBase, files, selectedModel);
      setPrediction(result);
      setActivePlane(result.preview_images[0]?.plane ?? "axial");
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Не удалось выполнить инференс");
    } finally {
      setIsRunning(false);
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-icon" aria-hidden="true">
            <Brain size={24} />
          </div>
          <div>
            <h1>Сегментация опухолей головного мозга</h1>
            <p>Демонстрационный прототип анализа МРТ</p>
          </div>
        </div>
        <div className="topbar-actions">
          <label className="api-field">
            <span>API</span>
            <input value={apiBase} onChange={(event) => setApiBase(event.target.value)} />
          </label>
          <button className="icon-button" type="button" onClick={refreshApiState} title="Обновить API">
            <RefreshCcw size={17} />
          </button>
          <StatusBadge state={apiStatus} />
        </div>
      </header>

      <section className="workspace-grid">
        <form className="panel upload-panel" onSubmit={handleSubmit}>
          <PanelTitle icon={<UploadCloud size={18} />} title="Входные данные" />
          <div className="upload-progress">
            <span>{completedUploads}/4 MRI</span>
            <div>
              <strong>{selectedModelInfo?.display_name ?? "Модель не выбрана"}</strong>
              <small>{selectedModelInfo?.weights_available ? "веса доступны" : "веса не найдены"}</small>
            </div>
          </div>
          <label className="field-label">
            Модель
            <select
              value={selectedModel}
              onChange={(event) => setSelectedModel(event.target.value)}
              disabled={!models.length}
            >
              {models.map((model) => (
                <option key={model.key} value={model.key}>
                  {model.display_name}
                </option>
              ))}
            </select>
          </label>
          <div className="upload-list">
            {MODALITIES.map((item) => (
              <FileInput
                key={item.key}
                title={item.title}
                description={item.description}
                file={files[item.key]}
                required={item.required}
                onChange={(event) => handleFileChange(item.key, event)}
              />
            ))}
          </div>
          {error && <div className="notice error">{error}</div>}
          <button
            className="primary-button"
            type="submit"
            disabled={isRunning || apiStatus !== "online" || !selectedModel}
          >
            <Play size={18} />
            {isRunning ? "Выполняется инференс" : "Запустить сегментацию"}
          </button>
        </form>

        <section className="panel result-panel">
          <PanelTitle icon={<FileScan size={18} />} title="Результат сегментации" />
          {prediction && selectedPreview ? (
            <>
              <div className="result-toolbar">
                <div>
                  <span className="muted">Prediction ID</span>
                  <strong>{prediction.prediction_id}</strong>
                </div>
                <div className="plane-tabs" role="tablist" aria-label="Плоскость просмотра">
                  {prediction.preview_images.map((image) => (
                    <button
                      key={image.plane}
                      className={image.plane === activePlane ? "active" : ""}
                      type="button"
                      onClick={() => setActivePlane(image.plane)}
                    >
                      {PLANE_LABELS[image.plane] ?? image.plane}
                    </button>
                  ))}
                </div>
              </div>
              <div className="image-compare">
                <PreviewCard
                  title="Исходный срез"
                  subtitle={`${selectedPreview.modality}, slice ${selectedPreview.slice_index}`}
                  src={selectedPreview.original_url}
                />
                <PreviewCard
                  title="Маска опухоли"
                  subtitle="наложение и контур"
                  src={selectedPreview.overlay_url}
                />
              </div>
              <div className="legend-row">
                <span><i className="legend yellow" /> TC / label 1</span>
                <span><i className="legend blue" /> WT edema / label 2</span>
                <span><i className="legend red" /> ET / label 4</span>
              </div>
              <div className="slice-notes">
                {selectedPreview.highlighted_labels.map((label) => (
                  <span key={label}>{label}</span>
                ))}
              </div>
            </>
          ) : (
            <EmptyState />
          )}
        </section>

        <aside className="panel research-panel">
          <PanelTitle icon={<FlaskConical size={18} />} title="Исследовательская сводка" />
          <MetricGrid prediction={prediction} />
          {prediction?.reference_metrics && <ReferenceMetrics metrics={prediction.reference_metrics} />}
          {prediction?.segmentation_url && (
            <a className="download-button" href={prediction.segmentation_url}>
              <Download size={17} />
              Скачать seg.nii.gz
            </a>
          )}
          <ExperimentTable experiments={experiments} />
        </aside>
      </section>
    </main>
  );
}

function PanelTitle({ icon, title }: { icon: ReactNode; title: string }) {
  return (
    <div className="panel-title">
      <span>{icon}</span>
      <h2>{title}</h2>
    </div>
  );
}

function StatusBadge({ state }: { state: "checking" | "online" | "offline" }) {
  const labels = {
    checking: "проверка",
    online: "API online",
    offline: "API offline",
  };
  return (
    <div className={`status-badge ${state}`}>
      <Server size={16} />
      {labels[state]}
    </div>
  );
}

function FileInput({
  title,
  description,
  file,
  required,
  onChange,
}: {
  title: string;
  description: string;
  file: File | null;
  required: boolean;
  onChange: (event: ChangeEvent<HTMLInputElement>) => void;
}) {
  return (
    <label className={`file-tile ${file ? "filled" : ""}`}>
      <input type="file" accept=".nii,.gz,.nii.gz" onChange={onChange} />
      <span className="file-token">{title}</span>
      <span className="file-copy">
        <strong>{file?.name ?? (required ? "Загрузить файл" : "Добавить при наличии")}</strong>
        <small>{description}</small>
      </span>
      {file && <CheckCircle2 size={18} />}
    </label>
  );
}

function PreviewCard({ title, subtitle, src }: { title: string; subtitle: string; src: string }) {
  return (
    <figure className="preview-card">
      <img src={src} alt={title} />
      <figcaption>
        <strong>{title}</strong>
        <span>{subtitle}</span>
      </figcaption>
    </figure>
  );
}

function EmptyState() {
  return (
    <div className="empty-state">
      <Activity size={32} />
      <h3>Ожидание результата</h3>
      <p>После запуска здесь появятся исходные срезы, маска опухоли и исследовательские метрики.</p>
    </div>
  );
}

function MetricGrid({ prediction }: { prediction: PredictionResponse | null }) {
  const stats = prediction?.voxel_statistics ?? {};
  return (
    <div className="metric-grid">
      <Metric label="WT voxels" value={stats.WT} />
      <Metric label="TC voxels" value={stats.TC} />
      <Metric label="ET voxels" value={stats.ET} />
      <Metric label="Модель" value={prediction?.model_used} />
    </div>
  );
}

function Metric({ label, value }: { label: string; value?: number | string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value ?? "—"}</strong>
    </div>
  );
}

function ReferenceMetrics({ metrics }: { metrics: Record<string, number> }) {
  return (
    <section className="reference-box">
      <h3>Сравнение с эталоном</h3>
      <div className="metric-grid compact">
        <Metric label="Mean Dice" value={formatMetric(metrics.mean_dice)} />
        <Metric label="Dice WT" value={formatMetric(metrics.dice_wt)} />
        <Metric label="Dice TC" value={formatMetric(metrics.dice_tc)} />
        <Metric label="Dice ET" value={formatMetric(metrics.dice_et)} />
      </div>
    </section>
  );
}

function ExperimentTable({ experiments }: { experiments: ExperimentSummary[] }) {
  const rows = experiments.slice(0, 5);
  return (
    <section className="experiment-table">
      <h3>Эксперименты</h3>
      {rows.length ? (
        <table>
          <thead>
            <tr>
              <th>Модель</th>
              <th>Mean</th>
              <th>WT</th>
              <th>TC</th>
              <th>ET</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((experiment) => (
              <tr key={experiment.key}>
                <td>
                  <strong>{shortExperimentName(experiment)}</strong>
                  <span>epoch {experiment.best_epoch ?? "—"}</span>
                </td>
                <td>{formatMetric(experiment.mean_dice)}</td>
                <td>{formatMetric(experiment.dice_wt)}</td>
                <td>{formatMetric(experiment.dice_tc)}</td>
                <td>{formatMetric(experiment.dice_et)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p className="muted">Сводки экспериментов пока не найдены.</p>
      )}
    </section>
  );
}

function formatMetric(value?: number | null): string {
  if (value === null || value === undefined) {
    return "—";
  }
  return value.toFixed(3);
}

function shortExperimentName(experiment: ExperimentSummary): string {
  if (experiment.experiment_name.includes("transfer-segresnet")) {
    return experiment.experiment_name.includes("refine") ? "SegResNet v2" : "SegResNet";
  }
  if (experiment.experiment_name.includes("baseline")) {
    return "3D U-Net";
  }
  return experiment.model_name || experiment.key;
}

export default App;
