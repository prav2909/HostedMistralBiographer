# HuggingFaceEmbedding is a wrapper class that will let us use pre-trained text embedding models from Hugging Face
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

"""
Quick summary of classes, they are also explained in  more detail later in the scripts
SimpleDirectoryReader: A smart reader class that needs a path to a directory and automatically spawns
 appropriate readers to read documents of different formats in that directory.
Settings: A wrapper class that acts as a config file for query and indexing class. These configurations are
 declared here with Settings and are globally available to each of these sub class/functions.
VectorStoreIndex: Stores data in form of indexes
VectorIndexRetriever: Retrive data based on index
RetrieverQueryEngine: An class that retuns index of data based on some rules like similarity
SimilarityPostprocessor: Class to calculate similarity score between query and documents in RAG corpus
"""
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _prompt_template_wo_context(query):

    intstructions_string = f"""You are PortfolioGPT, you have to answer queries about a person.
    If you do not have any information about the person, say that you do not know. Do not answer questions
    that are inappropriate in nature and be polite. Keep responses short.
    """
    prompt_template = lambda query: f'''[INST] {intstructions_string} \n{query} \n[/INST]'''

    return prompt_template(query)

def _prompt_template_w_context(context, query):
    prompt_template_w_context = lambda context, query: f"""You are PortfolioGPT, you have to answer queries about a person.
    If you do not have any information about the person, say that you do not know. Do not answer questions
    that are inappropriate in nature and be polite. Keep responses short.

    {context}
    Please respond to the following comment. Use the context above if it is helpful.

    {query}
    [/INST]
    """
    return prompt_template_w_context(context, query)

def _remove_strings_from_data(documents, STRINGS_TO_REMOVE):
    
    for doc in documents:
        for string in STRINGS_TO_REMOVE:
            if string in doc.text:
                documents.remove(doc)

    return documents

class MitralPlusRAG():
    def __init__(self, model_name='TheBloke/Mistral-7B-Instruct-v0.2-GPTQ', strings_to_remove=['']):
        self.settings = Settings
        self.settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.settings.llm = None # We do not want an LLama Index model, since we have a model from hugging face
        self.settings.chunk_size = 256 # What should be the token/per chunk in which the text corpus is divided into: set to 256 tokens
        self.settings.chunk_overlap = 25 # The overlap between two chunks, common tokens in two chunks
        documents = SimpleDirectoryReader("data").load_data()
        # self.STRINGS_TO_REMOVE = strings_to_remove
        documents = _remove_strings_from_data(documents, strings_to_remove)
        index = VectorStoreIndex.from_documents(documents, show_progress=True)
        self.top_k = 3
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=self.top_k,
        )
        self.query_engine = RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                    device_map="auto",
                                                    trust_remote_code=False,
                                                    revision="main")
        # model_name = "TheBloke/MistralLite-7B-GGUF"
        # self.model = AutoModelForCausalLM.from_pretrained("TheBloke/MistralLite-7B-GGUF", model_file="mistrallite.Q4_K_M.gguf", model_type="mistral", gpu_layers=50)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    
    def _return_without_RAG(self, query):

        prompt = _prompt_template_wo_context(query)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(input_ids=inputs["input_ids"].to(device), max_new_tokens=280)
        res = self.tokenizer.batch_decode(outputs)[0]
        return str(res.split('[/INST]')[-1])
        
    def _return_with_RAG(self, query):

        retreival_engine_response = self.query_engine.query(query)
        context = "Context:\n"
        for i in range(self.top_k):
            context = context + retreival_engine_response.source_nodes[i].text + "\n\n"
        prompt = _prompt_template_w_context(context, query)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(input_ids=inputs["input_ids"].to(device), max_new_tokens=280)
        res = self.tokenizer.batch_decode(outputs)[0]
        return str(res.split('[/INST]')[-1])

if __name__=="__main__":
    query = "Who is Praveen"
    runner = MitralPlusRAG()
    print("### Without RAG ###")
    res = runner._return_without_RAG(query)
    print(res)
    print("### With RAG ###")
    res = runner._return_with_RAG(query)
    print(res)
