from sentence_transformers import SentenceTransformer

# Загрузите модель один раз при наличии интернета
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
