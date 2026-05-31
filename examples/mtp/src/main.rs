//! MTP (Multi-Token Prediction) draft→verify→accept generation CLI.
//!
//! Thin command-line front end over the [`mtp`] library: it parses arguments,
//! loads a real Qwen3.6 MTP GGUF once, and drives [`mtp::mtp_generate`], which
//! creates a *target* context ([`LlamaContextType::Default`]) and a *draft*
//! context ([`LlamaContextType::Mtp`]) on the same model and runs the
//! speculative draft→verify→accept loop from `mtp-orchestration.md` to generate
//! text. It prints the decoded completion and the per-run acceptance stats
//! (`accepted`/`target_passes`/`emitted`) so the speedup signal — acceptance >
//! 0, i.e. fewer target forward passes than tokens emitted — is observable.
//!
//! [`LlamaContextType::Default`]: llama_cpp_2::context::params::LlamaContextType::Default
//! [`LlamaContextType::Mtp`]: llama_cpp_2::context::params::LlamaContextType::Mtp
//!
//! # Running
//!
//! Download a Qwen3.6 MTP GGUF (e.g. from `unsloth/Qwen3.6-35B-A3B-MTP-GGUF`)
//! and point the example at it:
//!
//! ```sh
//! # use an already-downloaded model
//! cargo run --release --example mtp -- local /path/to/qwen3.6-mtp.gguf
//!
//! # or download (and cache) from huggingface
//! cargo run --release --example mtp -- \
//!     hf-model unsloth/Qwen3.6-35B-A3B-MTP-GGUF Qwen3.6-35B-A3B-MTP-Q4_K_M.gguf
//! ```
//!
//! Pass `--prompt` to change the prompt and `--n-predict` to change how many
//! tokens to generate.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use hf_hub::api::sync::ApiBuilder;

use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{LlamaModel, Special};

use mtp::{mtp_generate, MtpParams};

#[derive(clap::Parser, Debug, Clone)]
struct Args {
    /// The path to the model
    #[command(subcommand)]
    model: Model,
    /// The prompt to seed the target context with.
    #[clap(short, long, default_value = "The quick brown fox")]
    prompt: String,
    /// The number of tokens to generate.
    #[clap(short, long, default_value_t = 64)]
    n_predict: usize,
    /// Disable offloading layers to the gpu
    #[cfg(any(feature = "cuda", feature = "vulkan"))]
    #[clap(long)]
    disable_gpu: bool,
}

#[derive(clap::Subcommand, Debug, Clone)]
enum Model {
    /// Use an already downloaded model
    Local {
        /// The path to the local GGUF model file.
        path: PathBuf,
    },
    /// Download a model from huggingface (or use a cached version)
    #[clap(name = "hf-model")]
    HuggingFace {
        /// the repo containing the model. e.g. `unsloth/Qwen3.6-35B-A3B-MTP-GGUF`
        repo: String,
        /// the model file name within the repo.
        model: String,
    },
}

impl Model {
    /// Convert the model selector to a local path - may download from huggingface.
    fn get_or_load(self) -> Result<PathBuf> {
        match self {
            Model::Local { path } => Ok(path),
            Model::HuggingFace { model, repo } => ApiBuilder::new()
                .with_progress(true)
                .build()
                .with_context(|| "unable to create huggingface api")?
                .model(repo)
                .get(&model)
                .with_context(|| "unable to download model"),
        }
    }
}

fn main() -> Result<()> {
    let Args {
        model,
        prompt,
        n_predict,
        #[cfg(any(feature = "cuda", feature = "vulkan"))]
        disable_gpu,
    } = Args::parse();

    let backend = LlamaBackend::init()?;

    // offload all layers to the gpu when a gpu backend is enabled
    let model_params = {
        #[cfg(any(feature = "cuda", feature = "vulkan"))]
        if !disable_gpu {
            LlamaModelParams::default().with_n_gpu_layers(1000)
        } else {
            LlamaModelParams::default()
        }
        #[cfg(not(any(feature = "cuda", feature = "vulkan")))]
        LlamaModelParams::default()
    };

    let model_path = model
        .get_or_load()
        .with_context(|| "failed to get model from args")?;

    // Load the model once; both contexts share it.
    let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
        .with_context(|| "unable to load model")?;

    let output = mtp_generate(&backend, &model, &prompt, n_predict, MtpParams::default())?;

    let text = model
        .tokens_to_str(&output.tokens, Special::Tokenize)
        .with_context(|| "failed to decode generated tokens")?;

    println!("{prompt}{text}");
    println!(
        "accepted={} target_passes={} emitted={}",
        output.accepted_total,
        output.target_passes,
        output.tokens.len()
    );

    Ok(())
}
