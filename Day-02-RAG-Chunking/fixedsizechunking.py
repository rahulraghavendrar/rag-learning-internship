from langchain_text_splitters import RecursiveCharacterTextSplitter
text = """
Internship duration is 2 months.

Interns receive a stipend of 15000 rupees.

Mentors conduct weekly reviews.

Certificates are issued upon completion.
"""
sizes=[20,50,100]
for size in sizes:
  print("Chunk size:",size)
  splitter=RecursiveCharacterTextSplitter(chunk_size=size,chunk_overlap=0)
  chunks=splitter.split_text(text)
  print("No of chunks is:",len(chunks))
  for chunk in chunks:
    print(chunk)
    print("\n")