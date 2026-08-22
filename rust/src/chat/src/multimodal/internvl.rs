// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! InternVL vision preprocessing and model spec for the Rust frontend.
//!
//! Ports vLLM's Python InternVL processing
//! (`vllm/transformers_utils/processors/internvl.py`) so the Rust frontend can
//! serve `InternVLChatModel` models:
//!
//! - dynamic tiling: pick the `(i, j)` grid with `i*j <= max_dynamic_patch`
//!   closest to the source aspect ratio, resize the image to
//!   `(448*i, 448*j)` (PIL-compatible bicubic), crop into 448x448 tiles, and
//!   append a 448x444 global thumbnail when there is more than one tile;
//! - per-tile CLIP normalization (OpenAI mean/std);
//! - prompt replacement: the `<img>` placeholder expands to
//!   `<img>` + `<IMG_CONTEXT>` * (`256 * num_patches`) + `</img>`;
//! - engine kwargs mirror the Python `MultiModalFieldConfig` declarations of
//!   `InternVLChatModel`: `pixel_values_flat` (flat tile concat),
//!   `image_num_patches` (batched, keep-on-cpu) and `image_token_id`
//!   (shared scalar, keep-on-cpu).

use std::collections::HashMap;

use image::{DynamicImage, GenericImageView};
use llm_multimodal::registry::RegistryResult;
use llm_multimodal::vision::transforms::{resize_bicubic_pil, to_tensor_and_normalize};
use llm_multimodal::{
    EncoderFieldLayouts, FieldLayout, Modality, ModelMetadata, ModelProcessorSpec,
    ModelSpecificValue, PreprocessedEncoderInputs, PromptReplacement, TokenId, TransformError,
    VisionPreProcessor,
};
use ndarray::Array4;
use serde_json::{Value, json};

/// Tile edge length. All InternVL checkpoints released to date use 448.
const IMAGE_SIZE: usize = 448;
/// ViT patch size. `(448 / 14)^2 = 1024` patches per tile.
const PATCH_SIZE: usize = 14;
/// Visual tokens per tile after the 2x2 pixel-shuffle downsample:
/// `1024 * (downsample_ratio=0.5)^2 = 256`.
const IMAGE_SEQ_LENGTH: usize =
    (IMAGE_SIZE / PATCH_SIZE) * (IMAGE_SIZE / PATCH_SIZE) / 4;
/// Upper bound on the dynamic tiling grid (`i * j`).
const MAX_DYNAMIC_PATCH: usize = 12;
/// Lower bound on the dynamic tiling grid (`i * j`).
const MIN_DYNAMIC_PATCH: usize = 1;
/// Whether to append a global 448x448 thumbnail when tiling.
const USE_THUMBNAIL: bool = true;

/// OpenAI CLIP normalization used by InternVL checkpoints.
const CLIP_MEAN: [f64; 3] = [0.48145466, 0.4578275, 0.40821073];
const CLIP_STD: [f64; 3] = [0.26862954, 0.26130258, 0.27577711];

/// Candidate tile grids with `min <= i*j <= max`, ordered by tile count.
///
/// Mirrors Python `get_internvl_target_ratios` (a set sorted by product);
/// the exact order of equal-product candidates only affects tie-breaking in
/// rare aspect-ratio edge cases and never changes output validity.
fn target_ratios(min_num: usize, max_num: usize) -> Vec<(usize, usize)> {
    let mut ratios: Vec<(usize, usize)> = (min_num..=max_num)
        .flat_map(|n| (1..=n).flat_map(move |i| (1..=n).map(move |j| (i, j))))
        .filter(|&(i, j)| min_num <= i * j && i * j <= max_num)
        .collect();
    ratios.sort_unstable_by_key(|&(i, j)| i * j);
    ratios.dedup();
    ratios
}

/// Pick the grid whose aspect ratio is closest to the source image's.
///
/// Port of Python `find_closest_aspect_ratio`, including the area-based
/// tie-break for equidistant candidates.
fn find_closest_aspect_ratio(
    aspect_ratio: f64,
    ratios: &[(usize, usize)],
    width: u32,
    height: u32,
    image_size: usize,
) -> (usize, usize) {
    let area = f64::from(width) * f64::from(height);
    let mut best_ratio = (1usize, 1usize);
    let mut best_ratio_diff = f64::INFINITY;
    for &(i, j) in ratios {
        let target_aspect = i as f64 / j as f64;
        let ratio_diff = (aspect_ratio - target_aspect).abs();
        if ratio_diff < best_ratio_diff {
            best_ratio_diff = ratio_diff;
            best_ratio = (i, j);
        } else if ratio_diff == best_ratio_diff
            && area > 0.5 * (image_size as f64) * (image_size as f64) * (i * j) as f64
        {
            best_ratio = (i, j);
        }
    }
    best_ratio
}

/// Number of 448x448 patches (tiles + optional thumbnail) for one image.
fn num_patches_for(width: u32, height: u32) -> usize {
    let ratios = target_ratios(MIN_DYNAMIC_PATCH, MAX_DYNAMIC_PATCH);
    let (grid_w, grid_h) =
        find_closest_aspect_ratio(f64::from(width) / f64::from(height), &ratios, width, height, IMAGE_SIZE);
    let mut patches = grid_w * grid_h;
    if USE_THUMBNAIL && patches > 1 {
        patches += 1;
    }
    patches
}

/// One normalized tile as `[3, 448, 448]`.
fn normalized_tile(tile: &DynamicImage) -> ndarray::Array3<f32> {
    to_tensor_and_normalize(
        &DynamicImage::ImageRgb8(tile.to_rgb8()),
        &CLIP_MEAN,
        &CLIP_STD,
    )
}

/// InternVL vision preprocessor.
///
/// Unlike the static default registry processors, this one carries the
/// tokenizer-resolved `<IMG_CONTEXT>` id so it can emit the shared
/// `image_token_id` kwarg expected by the Python `InternVLChatModel`.
pub struct InternVLVisionProcessor {
    ctx_token_id: TokenId,
}

impl InternVLVisionProcessor {
    pub fn new(ctx_token_id: TokenId) -> Self {
        Self { ctx_token_id }
    }

    /// Tile one image into normalized `[patches, 3, 448, 448]`.
    fn process_single_image(&self, image: &DynamicImage) -> Array4<f32> {
        let rgb = DynamicImage::ImageRgb8(image.to_rgb8());
        let (width, height) = rgb.dimensions();

        let ratios = target_ratios(MIN_DYNAMIC_PATCH, MAX_DYNAMIC_PATCH);
        let (grid_w, grid_h) = find_closest_aspect_ratio(
            f64::from(width) / f64::from(height),
            &ratios,
            width,
            height,
            IMAGE_SIZE,
        );
        let blocks = grid_w * grid_h;
        let target_width = IMAGE_SIZE * grid_w;
        let target_height = IMAGE_SIZE * grid_h;

        let mut resized = resize_bicubic_pil(&rgb, target_width as u32, target_height as u32);
        let mut tiles = Vec::with_capacity(blocks + 1);
        for index in 0..blocks {
            let x = (index % grid_w) * IMAGE_SIZE;
            let y = (index / grid_w) * IMAGE_SIZE;
            // `DynamicImage::crop` takes (x, y, width, height), unlike PIL's
            // (left, upper, right, lower) box.
            let tile = resized.crop(x as u32, y as u32, IMAGE_SIZE as u32, IMAGE_SIZE as u32);
            tiles.push(tile);
        }
        if USE_THUMBNAIL && blocks > 1 {
            // The thumbnail is built from the ORIGINAL image, not the tiled
            // one, matching Python `dynamic_preprocess_internvl`.
            tiles.push(resize_bicubic_pil(&rgb, IMAGE_SIZE as u32, IMAGE_SIZE as u32));
        }
        // Drop the grid-resized image before stacking to bound peak memory.
        drop(resized);

        let num_patches = tiles.len();
        let mut output = Array4::<f32>::zeros((num_patches, 3, IMAGE_SIZE, IMAGE_SIZE));
        for (index, tile) in tiles.iter().enumerate() {
            output
                .slice_mut(ndarray::s![index, .., .., ..])
                .assign(&normalized_tile(tile));
        }
        output
    }
}

impl VisionPreProcessor for InternVLVisionProcessor {
    fn default_mean(&self) -> [f64; 3] {
        CLIP_MEAN
    }

    fn default_std(&self) -> [f64; 3] {
        CLIP_STD
    }

    fn preprocess(
        &self,
        images: &[DynamicImage],
        _config: &llm_multimodal::PreProcessorConfig,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        if images.is_empty() {
            return Err(TransformError::InvalidShape {
                expected: "at least one image".to_string(),
                actual: vec![0],
            });
        }

        let mut patch_counts = Vec::with_capacity(images.len());
        let mut item_sizes = Vec::with_capacity(images.len());
        let mut flat_tiles =
            Array4::<f32>::zeros((0, 3, IMAGE_SIZE, IMAGE_SIZE));
        for image in images {
            let tiles = self.process_single_image(image);
            patch_counts.push(tiles.shape()[0]);
            item_sizes.push((image.width(), image.height()));
            // Concatenate along the tile axis across all images, producing
            // the flat layout Python calls `pixel_values_flat`.
            flat_tiles = ndarray::concatenate(
                ndarray::Axis(0),
                &[flat_tiles.view(), tiles.view()],
            )
            .expect("tile shapes are identical")
            .to_owned();
        }

        let feature_token_counts: Vec<usize> = patch_counts
            .iter()
            .map(|&patches| patches * IMAGE_SEQ_LENGTH)
            .collect();

        let num_images = images.len();
        let preprocessed = PreprocessedEncoderInputs::new(
            flat_tiles,
            feature_token_counts,
            item_sizes,
        )
        .with_extra(
            "image_num_patches",
            ModelSpecificValue::UintTensor {
                data: patch_counts.iter().map(|&n| n as u32).collect(),
                shape: vec![num_images],
            },
        )
        .with_extra("image_token_id", ModelSpecificValue::Int(i64::from(self.ctx_token_id)));
        Ok(preprocessed)
    }

    fn calculate_num_tokens(&self, width: u32, height: u32, _config: &llm_multimodal::PreProcessorConfig) -> usize {
        num_patches_for(width, height) * IMAGE_SEQ_LENGTH
    }

    fn model_name(&self) -> &'static str {
        "internvl"
    }
}

/// InternVL model spec: placeholder, prompt replacement and field layouts.
pub struct InternVLProcessorSpec;

/// Whether a model is an InternVL chat model, per its config or id.
pub(super) fn is_internvl(metadata: &ModelMetadata) -> bool {
    metadata.config_model_type() == Some("internvl_chat")
        || metadata
            .model_id
            .to_lowercase()
            .contains("internvl")
}

impl ModelProcessorSpec for InternVLProcessorSpec {
    fn name(&self) -> &'static str {
        "internvl"
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        is_internvl(metadata)
    }

    /// The template-visible placeholder. Python uses the multi-token string
    /// `<image>` and replaces it at string level; the Rust frontend needs a
    /// single-vocab-token marker, so we use `<img>` and re-emit it inside the
    /// replacement, which yields the identical final token sequence.
    fn placeholder_token(&self, _metadata: &ModelMetadata) -> RegistryResult<String> {
        Ok("<img>".to_string())
    }

    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        metadata.token_id("<IMG_CONTEXT>")
    }

    fn modality_limits(
        &self,
        _metadata: &ModelMetadata,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        Ok(HashMap::from([(Modality::Image, 4)]))
    }

    fn processor_kwargs(&self, _metadata: &ModelMetadata) -> RegistryResult<Value> {
        Ok(json!({}))
    }

    fn prompt_replacements(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        let start_id = metadata.token_id("<img>")?;
        let ctx_id = metadata.token_id("<IMG_CONTEXT>")?;
        let end_id = metadata.token_id("</img>")?;
        Ok(preprocessed
            .feature_token_counts
            .iter()
            .map(|&count| {
                let mut tokens = Vec::with_capacity(count + 2);
                tokens.push(start_id);
                tokens.extend(std::iter::repeat(ctx_id).take(count));
                tokens.push(end_id);
                PromptReplacement::sequence(Modality::Image, "<img>", tokens)
            })
            .collect())
    }

    /// `pixel_values_flat` is sliced per image by the `image_num_patches`
    /// sizes tensor, mirroring Python `MultiModalFieldConfig.flat_from_sizes`.
    fn encoder_field_layouts_for(&self, _modality: Modality) -> EncoderFieldLayouts {
        EncoderFieldLayouts::new(
            FieldLayout::flat("image_num_patches"),
            HashMap::from([("image_num_patches".to_string(), FieldLayout::Batched)]),
        )
    }

    fn keep_on_cpu_keys(&self) -> Vec<String> {
        vec![
            "image_num_patches".to_string(),
            "image_token_id".to_string(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image_of(width: u32, height: u32) -> DynamicImage {
        DynamicImage::new_rgb8(width, height)
    }

    #[test]
    fn seq_length_is_256() {
        assert_eq!(IMAGE_SEQ_LENGTH, 256);
    }

    #[test]
    fn patch_counts_match_python_tiling() {
        // 4:3 image picks the (4, 3) grid: 12 tiles + 1 thumbnail.
        assert_eq!(num_patches_for(800, 600), 13);
        // Exactly one tile: no thumbnail.
        assert_eq!(num_patches_for(448, 448), 1);
        // Square-ish large image picks (3, 3).
        assert_eq!(num_patches_for(1400, 1300), 10);
        // Wide image: (2,1) and (4,2) tie on aspect distance; the area
        // tie-break picks the larger grid, so 8 tiles + thumbnail.
        assert_eq!(num_patches_for(1708, 902), 9);
    }

    #[test]
    fn preprocess_builds_flat_tile_tensor() {
        let processor = InternVLVisionProcessor::new(151_667);
        let preprocessed = processor
            .preprocess(&[image_of(800, 600), image_of(448, 448)], &Default::default())
            .expect("preprocess succeeds");

        // 13 + 1 tiles flattened along dim 0.
        assert_eq!(preprocessed.encoder_input.shape(), &[14, 3, 448, 448]);
        assert_eq!(preprocessed.feature_token_counts, vec![13 * 256, 256]);
        assert_eq!(preprocessed.item_sizes, vec![(800, 600), (448, 448)]);

        match preprocessed.model_specific.get("image_num_patches") {
            Some(ModelSpecificValue::UintTensor { data, shape }) => {
                assert_eq!(data, &vec![13, 1]);
                assert_eq!(shape, &vec![2]);
            }
            _ => panic!("unexpected image_num_patches value"),
        }
        match preprocessed.model_specific.get("image_token_id") {
            Some(ModelSpecificValue::Int(value)) => assert_eq!(*value, 151_667),
            _ => panic!("unexpected image_token_id value"),
        }
    }
}

