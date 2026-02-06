from sentence_transformers import SentenceTransformer
from pathlib import Path
# Get the model ID from Hugging Face
MODEL_ID = 'Qwen/Qwen3-Embedding-0.6B'
OUT = Path(r'C:\GIT\Z_Master_Rag\Data\framemaker\mif_jsx/Qwen3-Reranker-0.6B')
print('Downloading model:', MODEL_ID)
model = SentenceTransformer(MODEL_ID)
print('Saving to:', OUT)
model.save(str(OUT))
print('Done')
