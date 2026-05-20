from langchain_huggingface import HuggingFaceEmbeddings

# Load embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Example text
text = "LangChain is used for building AI applications"

# Convert text into vector embedding
vector = embedding_model.embed_query(text)

print("Vector Length:", len(vector))
print(vector[:5])  # first 5 values