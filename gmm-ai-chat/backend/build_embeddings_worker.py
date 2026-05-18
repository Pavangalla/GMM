from .data_loader import build_database, build_embeddings

print("Starting database build...")
build_database()

print("Starting embeddings build...")
build_embeddings()

print("All done.")
