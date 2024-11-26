import os
import requests
from typing import List
from sklearn.neighbors import NearestNeighbors
from transformers import AutoTokenizer, AutoModel
import torch
from langgraph.graph import START, END, StateGraph
import ollama

# URL Loader
def fetch_url_content(urls: List[str]) -> List[str]:
    documents = []
    for url in urls:
        try:
            response = requests.get(url)
            response.raise_for_status()
            documents.append(response.text)
        except Exception as e:
            print(f"[ERROR] Error fetching {url}: {e}")
    return documents

# Query Ollama Model
def query_ollama(model: str, prompt: str, temperature: float = 0) -> str:
    response = ollama.Client().complete(model=model, prompt=prompt, temperature=temperature)
    return response.get("response", "")

# HuggingFace for the embedding stuff
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


def compute_embeddings(texts: List[str]) -> torch.Tensor:
    with torch.no_grad():
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings


# GraphState Class
class GraphState:
    def __init__(self, question: str):
        self.question: str = question
        self.documents: List[str] = []
        self.steps: List[str] = []
        self.answer: str = ""
        self.metadata: dict = {}

    def update(self, key: str, value):
        self.metadata[key] = value



# Define model and URLs
local_llm = "llama3.2"
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# Loads documents
docs_list = fetch_url_content(urls)
print("[DEBUG] Computing embeddings for loaded documents...")
doc_embeddings = compute_embeddings(docs_list).numpy()

# NearestNeighbors model
print("[DEBUG] Creating NearestNeighbors model...")
knn = NearestNeighbors(n_neighbors=4, metric="cosine")
knn.fit(doc_embeddings)

# Functions for each workflow step
def retrieve_documents(question: str, k: int = 4) -> List[str]:
    print(f"[DEBUG] Retrieving documents for question: {question}")
    query_embedding = compute_embeddings([question]).numpy()
    distances, indices = knn.kneighbors(query_embedding, n_neighbors=k)
    results = [docs_list[i] for i in indices[0]]
    print(f"[DEBUG] Retrieved documents: {results}")
    return results

def grade_documents(question: str, documents: List[str]) -> List[str]:
    relevant_docs = []
    for doc in documents:
        prompt = f"""You are a teacher grading a quiz. You will be given:
        Question: {question}
        Fact: {doc}
        You are grading RELEVANCE RECALL:
        A score of 1 means that ANY of the statements in the FACT are relevant to the QUESTION. 
        A score of 0 means that NONE of the statements in the FACT are relevant to the QUESTION. 
        1 is the highest (best) score. 0 is the lowest score you can give. 
        
        Explain your reasoning in a step-by-step manner. Ensure your reasoning and conclusion are correct. 
        
        Avoid simply stating the correct answer at the outset.
        
        Question: {question} \n
        Fact: \n\n {documents} \n\n
        
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
        """
        response = query_ollama(model=local_llm, prompt=prompt)
        print(f"[DEBUG] Grading response: {response}")
        if '"score": "yes"' in response:
            relevant_docs.append(doc)
    return relevant_docs

def generate_answer(question: str, documents: List[str]) -> str:
    doc_content = "\n\n".join(documents)
    prompt = f"""You are an assistant for question-answering tasks. 
    
        Use the following documents to answer the question. 
        
        If you don't know the answer, just say that you don't know. 
        
        Use three sentences maximum and keep the answer concise:
        Question: {question} 
        Documents: {documents} 
        Answer: 
        """
    response = query_ollama(model=local_llm, prompt=prompt)
    print(f"[DEBUG] Generated answer: {response}")
    return response


# Workflow Nodes
def retrieve(state: GraphState) -> GraphState:
    print(f"[DEBUG] Before retrieve: {vars(state)}")
    retrieved_docs = retrieve_documents(state.question)
    if not retrieved_docs:
        state.documents = ["No documents found."]
        state.metadata["retrieve"] = "No documents retrieved."
    else:
        state.documents = retrieved_docs
        state.metadata["retrieve"] = f"Retrieved {len(retrieved_docs)} documents."
    state.steps.append("retrieve_documents")
    print(f"[DEBUG] After retrieve: {vars(state)}")
    return state


def grade(state: GraphState) -> GraphState:
    print(f"[DEBUG] Before grade: {vars(state)}")
    if not state.documents or state.documents == ["No documents found."]:
        state.metadata["grade"] = "No documents to grade."
        state.steps.append("grade_documents")
        return state
    graded_docs = grade_documents(state.question, state.documents)
    if not graded_docs:
        state.documents = ["No relevant documents."]
        state.metadata["grade"] = "No relevant documents found."
    else:
        state.documents = graded_docs
        state.metadata["grade"] = f"Graded {len(graded_docs)} documents as relevant."
    state.steps.append("grade_documents")
    print(f"[DEBUG] After grade: {vars(state)}")
    return state


def generate(state: GraphState) -> GraphState:
    print(f"[DEBUG] Before generate: {vars(state)}")
    if not state.documents or state.documents == ["No relevant documents."]:
        state.answer = "No relevant information found."
        state.metadata["generate"] = "No answer generated."
    else:
        state.answer = generate_answer(state.question, state.documents)
        state.metadata["generate"] = "Answer generated successfully."
    state.steps.append("generate_answer")
    print(f"[DEBUG] After generate: {vars(state)}")
    return state


# Workflow Setup
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade", grade)
workflow.add_node("generate", generate)
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade")
workflow.add_edge("grade", "generate")
workflow.add_edge("generate", END)

# Print workflow structure
print("[DEBUG] Workflow nodes and edges:")
for node_name, node in workflow.nodes.items():
    print(f"Node: {node_name}, Function: {node}")
for edge in workflow.edges:
    print(f"Edge: {edge}")

# Execute the Workflow
example_question = "What are the types of agent memory?"
initial_state = GraphState(question=example_question)

try:
    print("[DEBUG] Starting workflow execution...")
    final_state = workflow.compile().invoke(initial_state)
    print("[DEBUG] Final State:", vars(final_state))
    print("Final Answer:", final_state.answer)
    print("Steps Taken:", final_state.steps)
    print("Metadata:", final_state.metadata)
except Exception as e:
    print(f"[ERROR] Workflow execution failed: {e}")
