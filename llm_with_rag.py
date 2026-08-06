"""
rag_utils.py

Once you've completed and tested Parts 1-5 in the assignment notebook,
paste your finished code into the matching sections below. This file is
what your Streamlit app (app.py) will import from.
"""
#Setup

import os
import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
CHROMADB_API_KEY = os.getenv("CHROMADB_API_KEY")
CHROMADB_TENANT = os.getenv("CHROMADB_TENANT")
CHROMADB_DB = os.getenv("CHROMADB_DB")

print("Imports OK" if DEEPSEEK_API_KEY else
      "Imports OK, but no DEEPSEEK_API_KEY found -- check your Colab secrets")
load_dotenv()

# --- Part 1: your knowledge base ---
documents = [
    "Coast to Kasbah is a premier design company specializing in custom logos, brand graphics, and high-performance websites.",
    "Logo design packages start at $500 and include 3 initial concepts, unlimited minor revisions, and full ownership rights with source files.",
    "Website design projects take 3 to 6 weeks depending on complexity, and include custom UI/UX design, responsive mobile layouts, and SEO optimization.",
    "Graphic design services include social media assets, marketing collateral, infographics, and custom data visualizations billed at $85 per hour or flat project rates.",
    "We accept payments via major credit cards, bank transfer (ACH/Wire), PayPal, and Stripe, with a standard 50% deposit required before project kickoff.",
    "Revisions and scope changes beyond the initial project agreement are billed at our standard hourly rate of $85 per hour.",
    "Client support is available via email at hello@coasttokasbah.com or scheduled phone calls Monday through Friday, 9 AM to 6 PM EST.",
    "All custom website builds include 30 days of complimentary post-launch support and maintenance to ensure smooth operation.",
    "Order cancellations can be requested within 48 hours of deposit payment for a full refund if design work has not commenced.",
    "Coast to Kasbah offers a 10% discount on bundled design packages when clients combine logo design with full website development."
]

assert len(documents) == 10, f"You need exactly 10 documents, you have {len(documents)}"
doc_ids = [f"doc_{i}" for i in range(len(documents))]
for i, doc in enumerate(documents):
    print(f"[{doc_ids[i]}] {doc}")


# --- Part 2: build the vector store ---
def build_vector_store(documents, doc_ids):
    """Create a ChromaDB collection and populate it with the given documents.

    Returns: (collection, vectorizer)
    """
    client = chromadb.CloudClient(
      api_key=CHROMADB_API_KEY,
      tenant=CHROMADB_TENANT,
      database=CHROMADB_DB
  )

    collection_name = "coast_to_kasbah_kb"

    # Ensure a clean slate: delete the collection if it exists
    try:
        client.delete_collection(name=collection_name)
        print(f"Deleted existing collection '{collection_name}'.")
    except Exception as e:
        print(f"Collection '{collection_name}' did not exist or could not be deleted: {e}")

    # Create a new, empty collection
    collection = client.create_collection(
        name=collection_name,
        embedding_function=None
    )

    vectorizer = TfidfVectorizer()
    embeddings_matrix = vectorizer.fit_transform(documents)
    doc_embeddings = embeddings_matrix.toarray().tolist()

    collection.add(embeddings=doc_embeddings, documents=documents, ids=doc_ids)

    return collection, vectorizer


collection, vectorizer = build_vector_store(documents, doc_ids)
print(f"Stored {collection.count()} documents in collection '{collection.name}'")

# --- Part 3: retrieval ---
def retrieve(collection, vectorizer, question, n_results=1):
    """Return the n_results documents most relevant to `question`."""

    #   1. Embed the question using vectorizer.transform([question])
    q_embedding = vectorizer.transform([question]).toarray().tolist()

    #   2. Query the collection with that embedding
    results = collection.query(
        query_embeddings=q_embedding,
        n_results=n_results
    )
    #   3. Return results["documents"][0]
    return results["documents"][0]


# Try it out
test_question = "How long will my design take?"
retrieved = retrieve(collection, vectorizer, test_question)
print(f"Question: {test_question}")
for doc in retrieved:
    print(" -", doc)

# --- Part 4: prompt building + generation ---
#Defines the system prompt
SYSTEM_PROMPT = (
    "You are a helpful, friendly, and professional customer service assistant for Coast to Kasbah. "
    "Coast to Kasbah is a design company that specializes in creative custom logos, branding graphics, and websites."
    "Answer the user's question using ONLY the provided context. "
    "If the answer cannot be determined from the context, state clearly that you do not know "
    "If you do not know, suggest contacting support at hello@coasttokasbah.com."
)

#Builds the prompt function for just the current turn
def build_prompt(question, retrieved_docs):
    """Build the CURRENT TURN's user-message content (context + question)."""

    context = "\n".join(f"- {doc}" for doc in retrieved_docs)
    prompt = f"Context:\n{context}\n\nQuestion: {question}"

    return prompt


# Generate answer function (calls deepseek API)
def generate_answer(messages):
   """Send a full list of conversation messages to DeepSeek and return the response text."""
   # Use the already defined DEEPSEEK_API_KEY variable
   if not DEEPSEEK_API_KEY:
       return "(No DEEPSEEK_API_KEY found -- check your Colab Secrets or .env file)"

   client = OpenAI(
       api_key=DEEPSEEK_API_KEY,
       base_url="https://api.deepseek.com"
   )

   response = client.chat.completions.create(
       model="deepseek-chat",  # or "deepseek-v4-flash" if specified by your assignment
       messages=messages
   )

   return response.choices[0].message.content

# Try it out (single turn, no history yet)
prompt = build_prompt(test_question, retrieved)
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": prompt},
]
answer = generate_answer(messages)
print(answer)

# --- Part 5: end-to-end chat ---
def chat(collection, vectorizer, question, history=None, n_results=3):
    """End-to-end RAG with conversation memory.

    `history` is an optional list of prior {"role": ..., "content": ...}
    messages (e.g. from st.session_state.messages), NOT including the
    current `question`. Passing `history` in is what makes follow-up
    questions like "how long does that take?" work.
    """
    # 1. Retrieve relevant docs for `question`
    retrieved_docs = retrieve(collection, vectorizer, question, n_results=n_results)

    # 2. Build the CURRENT TURN's user content
    current_turn_prompt = build_prompt(question, retrieved_docs)

    # 3. Assemble messages: [system message] + (history or []) + [current turn]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": current_turn_prompt})

    # 4. Call generate_answer() with full message list
    answer = generate_answer(messages)

    # 5. Return the answer
    return answer

# --- Test Part 5: Multi-Turn Conversation ---
print("\n" + "=" * 70)
print("MULTI-TURN TEST")

# Turn 1: First question
turn_1_question = "Do you offer logo design packages?"
turn_1_answer = chat(collection, vectorizer, turn_1_question)
print("Q1:", turn_1_question)
print("A1:", turn_1_answer)

# Store conversation history after Turn 1
history = [
    {"role": "user", "content": turn_1_question},
    {"role": "assistant", "content": turn_1_answer},
]

# Turn 2: Follow-up question relying on memory (doesn't explicitly state "logo design")
turn_2_question = "How much does that cost and what does it include?"
turn_2_answer = chat(collection, vectorizer, turn_2_question, history=history)
print("\nQ2:", turn_2_question)
print("A2:", turn_2_answer)
