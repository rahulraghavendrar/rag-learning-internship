from pypdf import PdfReader

# Load PDF
reader = PdfReader(
    "internship_handbook.pdf"
)

# Extract text
text = ""

for page in reader.pages:

    text += page.extract_text()

# Display extracted text
print("FULL DOCUMENT\n")

print(text)

print("\n" + "=" * 50)

# Convert text into words
words = text.split()

# Chunk size
chunk_size = 20

# Store chunks
chunks = []

# Create chunks
for i in range(
    0,
    len(words),
    chunk_size
):

    chunk_words = words[
        i:i + chunk_size
    ]

    chunk_text = " ".join(
        chunk_words
    )

    chunks.append(
        chunk_text
    )

# Display chunks
print("\nCHUNKS\n")

for index, chunk in enumerate(
    chunks
):

    print(
        f"\nChunk {index + 1}"
    )

    print(chunk)

    print("-" * 50)