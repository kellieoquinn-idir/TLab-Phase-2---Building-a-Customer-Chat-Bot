# Coast to Kasbah – Customer Service RAG Chatbot

This project is a Retrieval-Augmented Generation (RAG) customer service chatbot built for **Coast to Kasbah**, a design studio specializing in custom logos, brand graphics, and web development.

The chatbot leverages vector search to retrieve relevant company policy and service information from a knowledge base, providing accurate, context-aware answers to user inquiries while maintaining conversational memory.

---

## **Features**

* **Custom Knowledge Base:** Contains structured document embeddings covering service offerings, pricing, turnarounds, payment methods, and support policies.
* **Vector Search Retrieval:** Uses `TfidfVectorizer` and `ChromaDB` (Cloud Client) to retrieve top matching knowledge base entries for user prompts.
* **LLM Integration:** Connects to DeepSeek API (`deepseek-chat`) via the OpenAI SDK for natural language response generation.
* **Multi-Turn Conversation Memory:** Handles follow-up questions seamlessly by passing full chat history alongside newly retrieved context.
* **Strict Guardrails:** Configured via system prompts to answer strictly within provided context and direct users to official support (`hello@coasttokasbah.com`) when information is missing.

---

## **Repository Structure**

* **`llm_with_rag.py`**: Core backend module containing vector store setup, retrieval logic, prompt construction, and the end-to-end `chat()` pipeline.
* **`app.py`**: Streamlit web interface for interactive customer chat.
* **`customer_service_rag_assignment[1].ipynb` / `ChatBot.ipynb**`: Jupyter notebooks detailing step-by-step development and testing.

---

## **Environment Setup**

### **1. Prerequisites**

Ensure you have Python 3.9+ installed along with the required dependencies:

```bash
pip install chromadb scikit-learn python-dotenv openai streamlit

```

### **2. Environment Variables**

Create a `.env` file in the root directory and add your API keys:

```env
DEEPSEEK_API_KEY=your_deepseek_api_key
CHROMADB_API_KEY=your_chromadb_api_key
CHROMADB_TENANT=your_chromadb_tenant_id
CHROMADB_DB=your_chromadb_database_name

```

---

## **Usage**

### **Run the Backend Script**

To test vector store initialization and multi-turn chat execution in the console:

```bash
python llm_with_rag.py

```

### **Launch the Web App**

To start the Streamlit chat application:

```bash
streamlit run app.py

```

---

## **How It Works**

1. **Vector Store Initialization:** `build_vector_store()` initializes a ChromaDB collection, transforms knowledge base documents using TF-IDF, and stores the resulting embeddings.
2. **Context Retrieval:** `retrieve()` embeds incoming user questions and queries ChromaDB for the $N$ most relevant knowledge base documents.
3. **Prompt Generation:** `build_prompt()` formats the retrieved context alongside the user's question.
4. **Response Generation:** `chat()` combines system guardrails, conversation history, and current context into an API payload sent to DeepSeek.
