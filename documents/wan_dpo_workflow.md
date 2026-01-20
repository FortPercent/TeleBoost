```mermaid
flowchart TD
  A[Load metadata list\nteletron/datasets/dpo_dataset.py: UnifiedDataset.load_metadata] --> 
  B[Fetch raw sample dict\nteletron/datasets/dpo_dataset.py: UnifiedDataset.__getitem__] --> 
  C{Iterate chosen/rejected keys\nteletron/datasets/dpo_dataset.py: UnifiedDataset.__getitem__}

  C --> D[Decode by extension\nteletron/datasets/dpo_dataset.py: UnifiedDataset.default_video_operator\n→ RouteByExtensionName/LoadVideo/LoadGIF/LoadImage]
  D --> E[Resize + crop frames\nteletron/datasets/dpo_dataset.py: ImageCropAndResize]
  E --> F[Build branch dict\nteletron/datasets/dpo_dataset.py: UnifiedDataset.__getitem__]
  F --> G[Inject video meta\nteletron/datasets/dpo_dataset.py: UnifiedDataset.inject_video_meta]

  G --> H[InjectRawFirstImageFromVideo\nteletron/datasets/transform/video_transform.py: InjectRawFirstImageFromVideo.__call__]
  H --> I[PreprocessVideoToTensor\nteletron/datasets/transform/video_transform.py: PreprocessVideoToTensor.__call__]
  I --> J[InjectImagesFromVideoTensor\nteletron/datasets/transform/video_transform.py: InjectImagesFromVideoTensor.__call__]
  J --> K[InjectPromptToTopLevel\nteletron/datasets/transform/prompt_transform.py: InjectPromptToTopLevel.__call__]
  K --> L[PackInputsNoResize\nteletron/datasets/transform/formatting.py: PackInputsNoResize.__call__]
  L --> M[Output branches dict\nteletron/datasets/dpo_dataset.py: UnifiedDataset.__getitem__]

```