export type SystemRole = 'system_admin' | 'user'
export type ProjectRole = 'project_owner' | 'researcher' | 'viewer'

export interface User {
  id: string
  email: string
  name: string
  system_role: SystemRole
  is_active: boolean
}

export interface ProjectMember {
  id: string
  user_id: string
  email: string
  name: string
  role: ProjectRole
}

export interface Project {
  id: string
  name: string
  description: string
  created_by: string
  created_at: string
  updated_at: string
  current_user_role: ProjectRole | null
  members: ProjectMember[]
}

export type ResourceVersionStatus = 'processing' | 'ready' | 'failed'
export type DatasetFormat = 'yolo' | 'coco' | 'classification' | 'generic'

export interface DatasetVersion {
  id: string
  dataset_id: string
  version: number
  status: ResourceVersionStatus
  format: DatasetFormat
  source_filename: string
  sha256: string
  archive_size: number
  extracted_size: number
  file_count: number
  error_message: string
  created_at: string
  ready_at: string | null
}

export interface Dataset {
  id: string
  project_id: string
  name: string
  description: string
  created_at: string
  versions: DatasetVersion[]
}

export interface CodeVersion {
  id: string
  code_package_id: string
  version: number
  status: ResourceVersionStatus
  source_filename: string
  sha256: string
  archive_size: number
  extracted_size: number
  file_count: number
  default_workdir: string
  default_entrypoint: string
  error_message: string
  created_at: string
  ready_at: string | null
}

export interface CodePackage {
  id: string
  project_id: string
  name: string
  description: string
  created_at: string
  versions: CodeVersion[]
}

export type TrainingRunStatus =
  | 'draft'
  | 'queued'
  | 'preparing'
  | 'starting'
  | 'running'
  | 'cancel_requested'
  | 'stopping'
  | 'stopped'
  | 'succeeded'
  | 'failed'
export type GpuModelPolicy = 'any' | 'rtx_3090' | 'rtx_4090' | 'tesla_t4'

export interface TrainingTemplate {
  id: string
  key: string
  name: string
  description: string
  default_runtime_image_id: string
  parameter_schema: Record<string, { type: string; default: string | number; minimum?: number }>
}

export interface RuntimeImage {
  id: string
  name: string
  image: string
  digest: string
  framework: string
}

export interface GpuDevice {
  id: string
  uuid: string
  index: number
  model: GpuModelPolicy
  memory_gb: number
  is_enabled: boolean
  allocated_run_id: string | null
}

export interface RunEvent {
  id: string
  event_type: string
  message: string
  created_at: string
}

export interface TrainingRun {
  id: string
  project_id: string
  name: string
  template_id: string
  template_name: string
  dataset_version_id: string
  dataset_label: string
  code_version_id: string
  code_label: string
  runtime_image_id: string
  runtime_image_name: string
  status: TrainingRunStatus
  parameters: Record<string, unknown>
  command: string[]
  requested_gpu_count: number
  requested_gpu_model: GpuModelPolicy
  allocated_gpus: string[]
  priority: number
  execution_id: string | null
  exit_code: number | null
  failure_reason: string
  created_by: string
  created_at: string
  queued_at: string | null
  started_at: string | null
  finished_at: string | null
  events: RunEvent[]
}

export interface MetricPoint {
  id: string
  key: string
  step: number
  value: number
  created_at: string
}

export interface RunArtifact {
  id: string
  run_id: string
  artifact_type: 'checkpoint' | 'report' | 'export' | string
  name: string
  size_bytes: number
  sha256: string
  details: Record<string, unknown>
  created_at: string
}

export type ModelStage = 'candidate' | 'production' | 'archived'
export type ExportStatus = 'pending' | 'succeeded' | 'failed'

export interface ExportJob {
  id: string
  model_version_id: string
  format: string
  status: ExportStatus
  artifact_id: string | null
  validation_status: string
  max_abs_diff: number | null
  error_message: string
  created_at: string
  finished_at: string | null
}

export interface ModelVersion {
  id: string
  version: number
  source_run_id: string
  source_run_name: string
  source_artifact_id: string
  source_artifact_name: string
  stage: ModelStage
  format: string
  metrics: Record<string, number>
  created_at: string
  exports: ExportJob[]
}

export interface RegisteredModel {
  id: string
  project_id: string
  name: string
  description: string
  created_by: string
  created_at: string
  versions: ModelVersion[]
}
