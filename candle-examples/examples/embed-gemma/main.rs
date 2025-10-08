use std::time::Instant;
use candle::{Device, DType};
use candle_transformers::models::embed_gemma::Gemma3Embedder;
use std::fs::File;
use std::io::Write;

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b)
}

fn main() -> anyhow::Result<()> {
    // Run on CPU for now
    let device = Device::Cpu;

    // Load EmbeddingGemma-300m from Hugging Face Hub
    let embedder = Gemma3Embedder::new(
        "google/embeddinggemma-300m",
        &device,
        DType::F32,
    )?;

    // ---- Single sentence ----
    let start = Instant::now();
    let emb = embedder.embed("hello world")?;
    let elapsed = start.elapsed();
    println!("Single embedding took {:?}", elapsed);
    println!("Single embedding dim = {}", emb.len());

    // ---- Batch of sentences ----
    let texts = vec![
        "the cat sat on the mat".to_string(),
        "dogs are cute".to_string(),
    ];
    let batch_emb = embedder.embed_batch(texts)?;
    println!("Batch size = {}", batch_emb.len());
    println!("Each embedding dim = {}", batch_emb[0].len());

    let emb = embedder.embed("The quick brown fox jumps over the lazy dog")?;
    let mut file = File::create("embedding.json")?;
    writeln!(file, "{}", serde_json::to_string(&emb).unwrap())?;

    let pairs = vec![
    ("A man is eating food", "Someone is having a meal"),   // similar
    ("The cat is sleeping on the bed", "The dog is barking outside"), // dissimilar
    ("I love pizza", "Pizza is my favorite food"),          // similar
    ("She is reading a book", "He is playing football"),    // dissimilar
    ];

    for (a, b) in pairs {
        let emb_a = embedder.embed(a)?;
        let emb_b = embedder.embed(b)?;

        let sim = cosine_similarity(&emb_a, &emb_b);
        println!("\"{}\" vs \"{}\" => similarity {:.3}", a, b, sim);
    }

    let sentences = vec![
    "The cat sits outside",
    "A man is playing guitar",
    "I love pasta",
    "The new movie is awesome",
    "The cat plays in the garden",
    "A woman watches TV",
    "The new film is fantastic",
    "Do you like pizza?",
    ];

    let embeddings = embedder.embed_batch(sentences.iter().map(|s| s.to_string()).collect())?;

    for i in 0..sentences.len() {
        for j in (i + 1)..sentences.len() {
            let sim = cosine_similarity(&embeddings[i], &embeddings[j]);
            if sim > 0.6 {
                println!("Sim {:.3}: '{}' <-> '{}'", sim, sentences[i], sentences[j]);
            }
        }
    }

    

    Ok(())
}
