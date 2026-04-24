export type HealthResponse = {
  status: string;
  available_models: string[];
};

export type ModelInfo = {
  key: string;
  name: string;
  display_name: string;
  roi_size: number[];
  weights_available: boolean;
};

export type ExperimentSummary = {
  key: string;
  experiment_name: string;
  model_name: string;
  best_epoch: number | null;
  mean_dice: number | null;
  dice_tc: number | null;
  dice_wt: number | null;
  dice_et: number | null;
  checkpoint_available: boolean;
};

export type PreviewImage = {
  plane: "axial" | "coronal" | "sagittal" | string;
  slice_index: number;
  modality: string;
  original_path: string;
  original_url: string;
  overlay_path: string;
  overlay_url: string;
  highlighted_labels: string[];
};

export type PredictionResponse = {
  prediction_id: string;
  model_used: string;
  segmentation_path: string;
  segmentation_url: string;
  voxel_statistics: Record<string, number>;
  preview_images: PreviewImage[];
  reference_metrics: Record<string, number> | null;
};

export type UploadFiles = {
  t1: File | null;
  t1ce: File | null;
  t2: File | null;
  flair: File | null;
  referenceSeg: File | null;
};

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const payload = await response.json();
      detail = payload.detail ?? JSON.stringify(payload);
    } catch {
      detail = await response.text();
    }
    throw new Error(`HTTP ${response.status}: ${detail}`);
  }
  return response.json() as Promise<T>;
}

export async function fetchHealth(apiBase: string): Promise<HealthResponse> {
  return requestJson<HealthResponse>(`${apiBase}/health`);
}

export async function fetchModels(apiBase: string): Promise<ModelInfo[]> {
  const payload = await requestJson<{ models: ModelInfo[] }>(`${apiBase}/models`);
  return payload.models;
}

export async function fetchExperiments(apiBase: string): Promise<ExperimentSummary[]> {
  const payload = await requestJson<{ experiments: ExperimentSummary[] }>(
    `${apiBase}/experiments`,
  );
  return payload.experiments;
}

export async function runPrediction(
  apiBase: string,
  files: UploadFiles,
  model: string,
): Promise<PredictionResponse> {
  const formData = new FormData();
  if (!files.t1 || !files.t1ce || !files.t2 || !files.flair) {
    throw new Error("Загрузите все четыре MRI-модальности.");
  }
  formData.append("t1", files.t1);
  formData.append("t1ce", files.t1ce);
  formData.append("t2", files.t2);
  formData.append("flair", files.flair);
  formData.append("model", model);
  if (files.referenceSeg) {
    formData.append("reference_seg", files.referenceSeg);
  }
  return requestJson<PredictionResponse>(`${apiBase}/predict`, {
    method: "POST",
    body: formData,
  });
}
