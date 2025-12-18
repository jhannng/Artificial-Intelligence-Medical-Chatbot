### Research Summary

<hr />

**<span style="color: #A1AEB1;">Retrieval-Augmented Medical Chatbot Pipeline Architecture</span>**

| Module                           | Description                                                                                           | Inputs               | Outputs                           |
| -------------------------------- | ----------------------------------------------------------------------------------------------------- | -------------------- | --------------------------------- |
| **Document Ingestion**           | Loads documents, extracts text, filenames, timestamps, and page-level metadata.                       | PDF / DOCX / HTML    | Clean Text + Metadata             |
| **Preprocessing & Chunking**     | Normalizes text, removes boilerplate, creates overlapping chunks with rich metadata.                  | Page-level Text      | Text Chunks + Metadata            |
| **Embedding & Vectorization**    | Converts chunks to embeddings using OpenAI Embedding model, then stores in vector database.           | Text Chunks          | Vector Store Entries              |
| **Retriever & Re-Ranking**       | Performs semantic search, applies metadata filters, reranks results using cross-encoder.              | Query + Vector Store | Top-K Passages + Relevance Scores |
| **RAG Prompting**                | Constructs RAG prompt with system rules, retrieved passages, citations, and the user query.           | Top-k Passages       | Prompt for LLM                    |
| **LLM Generation**               | Produces draft medical response with citations and confidence.                                        | Structured Prompt    | Generated Answer                  |
| **Safety & Clinical Guardrails** | Checks for hallucination, unsafe content, missing citations, or clinical risks; may trigger fallback. | Generated Answer     | Safe Final Answer                 |
| **Audit Logging & Monitoring**   | Logs queries, retrievals, sources, and model responses for compliance.                                | System Events        | Audit Logs                        |
| **Evaluation & Feedback Loop**   | Measures accuracy, safe-fail rates, and retrieval quality to improve pipeline.                        | Logs + Responses     | Metrics & Model Improvements      |

<br />

<div style="text-align: justify;">

<span style="color: #D0312D;">**Note: The Retrieval-Augmented Generation (RAG) pipeline is currently being implemented as an early prototype. It does not yet contain every element specified in the architecture above, even though it illustrates the essential workflow from document ingestion to response generation. Later stages of development will include features like PHI de-identification, safety and compliance layers, improved retrieval optimisation, and thorough assessment systems.**</span>

<hr />

</div>

**<span style="color: #A1AEB1;">Retrieval-Augmented Generation Framework Research Summary</span>**

<div style="text-align: justify;">

Throughout the research into open-source Retrieval-Augmented Generation (RAG) frameworks, I evaluated several leading tools including LangChain, LlamaIndex, Haystack, FlexRAG, and UltraRAG based on factors such as ecosystem maturity, retrieval performance, orchestration flexibility, integration support, scalability, and suitability for medical domain requirements.

**LlamaIndex** stood out for its data-centric architecture, offering efficient document ingestion, structured indexing, and advanced query engines such as Tree and Graph indexes. It is excellent for rapid prototyping and scenarios where flexible data connectors are needed. However, it provides less orchestration control for multi-step conversational flows or agent-like reasoning.

**Haystack** demonstrated strengths in production-grade pipelines, robust hybrid retrieval (sparse + dense), and a modular architecture that fits enterprise settings well. Its pipeline API is powerful but can be heavier to configure and less ideal for agent-based interactions or dynamic tool-calling workflows.

**FlexRAG** and **UltraRAG** offer innovative retrieval strategies and automation capabilities. FlexRAG focuses on dynamic retrieval orchestration, while UltraRAG emphasizes pipeline optimization and evaluation tooling. Despite their promise, both frameworks are still relatively new, with smaller ecosystems, limited documentation, and fewer real-world examples.

<span style="color: #74B72E;">**LangChain** turned out to be the most sensible and adaptable alternative for this project when all the choices had been evaluated. It provides flexible workflow orchestration, strong support for tools, agents, and memory components, and broad integration with vector stores and LLM providers. LangChain also enables large-scale ingestion, distributed processing, detailed control over embedding/ vector store layers, and excellent OpenAI integration, which is important for medical RAG systems that require reliability and strict compliance handling. While LangChain may not include the most optimized retriever internally, its compatibility with external retrieval engines makes it the best fit for building a medically safe, extensible, and future-proof RAG-based medical chatbot.</span>

<hr />

</div>

**<span style="color: #A1AEB1;">Retrieval-Augmented Generation Large Language Model (LLM) Research Summary</span>**

<div style="text-align: justify;">

Large Language Models (LLM) service provider is a critical component for a medical Retrieval-Augmented Generation (RAG) system, as it directly impacts reasoning accuracy, safety alignment, latency, cost efficiency, and compliance readiness. Through this research phase, three major LLM service providers were evaluated which are OpenAI, Google Gemini, and Anthropic (Claude). The evaluation criteria included medical reasoning capability, hallucination control, safety mechanisms, ecosystem maturity, tooling support, and integration reliability with LangChain.

**OpenAI** currently offers the most mature and production-ready ecosystem for medical-adjacent applications. Its models demonstrate strong clinical language understanding, robust multi-step reasoning, and consistent performance across long-context inputs. OpenAI provides fine-grained system prompts, function calling, and strict safety policies, all of which are essential for medical advice scenarios where factual accuracy and response control are paramount. Additionally, OpenAI models integrate seamlessly with LangChain, offering first-class support for Retrieval-Augmented Generation (RAG) pipelines, structured outputs, and tool invocation.

**Google Gemini** models show strong performance in multimodal tasks and general reasoning, benefiting from Google’s extensive knowledge graph and search infrastructure. However, Gemini’s medical reasoning consistency and controllability remain less predictable in comparison to OpenAI, particularly for long-form clinical explanations. Integration with LangChain is improving but still less mature, and advanced prompt control mechanisms are more limited.

**Anthropic’s Claude** models are well-known for their safety-first design and strong summarization capabilities. Claude excels at reading long documents and generating coherent summaries, which is useful for medical literature analysis. However, Claude’s response style can be overly cautious or verbose, and its integration options for Retrieval-Augmented Generation (RAG) orchestration and tool-calling remain more restrictive compared to OpenAI.

<span style="color: #74B72E;">Based on this evaluation, **OpenAI** was selected as the LLM service provider for this medical chatbot project. Its superior reasoning consistency, advanced prompt and tool control, tight LangChain integration, and proven reliability in safety-critical domains make it the most suitable choice for Retrieval-Augmented Generation (RAG) in medical applications. While Google Gemini and Anthropic Claude offer compelling strengths in specific areas, OpenAI currently provides the best balance between accuracy, safety, developer productivity, and ecosystem maturity.</span>

<hr />

</div>

**<span style="color: #A1AEB1;">Document Ingestion (Document Loader)</span>**

<div style="text-align: justify;">

Through the research phase, I noticed that medical dataset often come in multiple formats such as PDF guidelines, DOCX clinical reports, TXT notes, HTML webpages, and more, which implies that each file type requires different preprocessing steps, and handling them individually can introduce errors, inconsistencies, and maintenance challenges. Therefore, a unified approach was necessary to streamline document loading while ensuring that all formats are processed correctly and consistently, and **Document Loader Factory** was implemented to address this need.

In general, it abstracts away file-specific parsing logic while ensuring that every document is transformed into a standardised structure suitable for chunking, embedding, and retrieval. This makes the overall Retrieval-Augmented Generation (RAG) pipeline more robust, extensible, and easier to maintain. Overall, the key reasons for implementing it is to ensure various file formats and future formats are process realiably, extensibility, consistent downstream processing, cleaner architecture, and reusability. Meanwhile, I believe that it is a foundational component that prepares unstructured medical data for high-quality retrieval in the Retrieval-Augmented Generation (RAG) medical chatbot.

</div>

**<span style="color: #A1AEB1;">Preprocessing & Chunking</span>**

<div style="text-align: justify;">

In comparison to more naive or fixed-window chunking methods, the **RecursiveCharacterTextSplitter** produces segments that better preserve semantic continuity, reduce unnatural text breakpoints, and align effectively with the processing requirements of downstream embedding models. Its recursive, rule-based splitting mechanism enables the generation of structurally coherent chunks while maintaining compatibility with LangChain’s document processing pipeline.

<span style="color: #74B72E;">Based on the evaluation across multiple chunking strategies applied to medical text corpora, a configuration of **500-character chunk size** with a **50-character overlap** emerged as the most suitable for this Retrieval-Augmented Generation (RAG) implementation. A chunk size of approximately 500 characters provides a strong balance between semantic density and processing efficiency, typically encapsulating one to two clinically meaningful paragraphs. This reduces the likelihood of fragmenting medically significant concepts, thereby improving embedding representativeness and enhancing retrieval precision.</span>

The 50-character overlap further addresses boundary effects by ensuring that clinical statements or sentence fragments located near chunk edges are preserved across adjacent segments. This approach minimizes semantic loss without introducing excessive redundancy in the vector store.

This chunking configuration is in line with best practices for Retrieval-Augmented Generation (RAG) in the medical field since it provides an efficient trade-off that requires preserving semantic coherence, optimising embedding quality, and facilitating effective storage and retrieval.

<hr />

</div>

**<span style="color: #A1AEB1;">Embedding Model</span>**

<div style="text-align: justify;">

As aforementioned, OpenAI’s services were selected for both embedding generation and Large Language Model (LLM) due to their demonstrated performance, operational stability, and seamless compatibility with the LangChain framework. Thus, the comparative analysis concentrates on three embedding models within the OpenAI ecosystem including text-embedding-3-small, text-embedding-3-large, and text-embedding-ada-002, and I will evaluate their relative effectiveness and suitability for downstream Retrieval-Augmented Generation (RAG) tasks.

The newest ligtweight embedding model from OpenAI, **text-embedding-3-small** model is intended to provide cutting-edge semantic performance with a smaller computing footprint. It provides significantly higher accuracy than older models despite operating at a lower dimensionality, enabling more efficient FAISS indexing and faster similarity search. Empirical testing indicates that this model achieves strong semantic alignment for medical terminology, clinical statements, and multi-sentence medical narratives, making it suitable for large-scale ingestion pipelines.

Additionally, **text-embedding-3-large** model is engineered for high-precision retrieval in scenarios where semantic fidelity and recall are prioritized over computational cost. It consistently yields stronger performance in tasks involving long-form medical documents, subtle clinical distinctions, and multi-hop medical reasoning. This makes it particularly valuable for safety-sensitive or high-recall applications such as diagnostic guidance, literature retrieval, and guideline summarization. However, the larger vector dimensionality increases memory usage and index size.

Compared with both text-embedding-3 series, the predecessor **text-embedding-ada-002** model previously served as OpenAI’s standard embedding solution. While ada-002 remains functional, it generally exhibits lower semantic resolution, weaker contextual understanding, and higher variance in retrieval quality, especially for domain-specific text such as medical literature and clinical guidelines. Its performance gap is evident across semantic similarity benchmarks and multi-paragraph retrieval tasks. Consequently, ada-002 is no longer preferred for systems requiring high accuracy or domain sensitivity.

<span style="color: #74B72E;">The empirical evaluation revealed that **text-embedding-3-small** was chosen for this study because it strikes the best possible balance between cost-effectiveness, computational efficiency, and semantic quality. It minimises index size and offers a high enough retrieval accuracy for medical-domain RAG tasks, making it appropriate for scale deployment and iterative prototyping. text-embedding-3-large is still a feasible upgrade path for situations needing maximum recall or managing intricate clinical narratives. Despite being compatible with any Large Language Model (LLM), OpenAI embeddings perform best when combined with OpenAI LLMs because of cross-component optimisation, shared semantic assumptions, and architectural alignment.</span>

<hr />

</div>

**<span style="color: #A1AEB1;">Vector Database</span>**

<div style="text-align: justify;">

Throughout the development of the Retrieval-Augmented Generation (RAG) pipeline, several vector databases and Approximate Nearest Neighbor (ANN) indexing frameworks were evaluated, including Pinecone, Weaviate, Milvus, ChromaDB, and HNSWlib, in addition to FAISS. Each solution offers a different balance of performance, scalability, operational complexity, and integration support.

**Pinecone** provides a fully managed, cloud-native vector database with strong scalability guarantees, low-latency retrieval, and robust metadata filtering. It is well suited for large-scale production deployments where operational overhead must be minimized. However, Pinecone introduces vendor lock-in, recurring infrastructure costs, and limited transparency into index configuration, which reduces flexibility for experimental or research-driven RAG development.

**Weaviate** offers an open-source, schema-driven vector database that supports hybrid (dense + sparse) search, modular storage backends, and both local and cloud deployments. While feature-rich, its schema-first design and operational footprint increase configuration complexity, which can slow down iteration cycles in small to medium-scale projects.

**Milvus**, designed for distributed vector storage at scale, demonstrates excellent performance for large datasets and production-grade workloads. However, its operational complexity which often requiring container orchestration, specialized storage, and dedicated cluster management makes it heavyweight for lightweight Retrieval-Augmented Generation (RAG) prototypes or academic experimentation.

**HNSWlib**, though extremely fast and easy to use, is limited by its in-memory design and lack of broader database features such as persistence, filtering, or integrated metadata search, making it less suitable as a standalone vector database in structured Retrieval-Augmented Generation (RAG) pipelines.

**FAISS** is a highly optimized vector similarity library offering fine-grained control over Approximate Nearest Neighbor (ANN) algorithms, GPU acceleration, and excellent retrieval performance. It is widely used in research and benchmarking environments. However, FAISS functions as a low-level indexing library rather than a complete vector database. As a result, features such as persistent storage, metadata filtering, versioning, and document lifecycle management must be implemented externally.

**ChromaDB** is a developer-oriented vector database explicitly designed for LLM-powered applications and Retrieval-Augmented Generation (RAG) workflows. Unlike FAISS or HNSWlib, Chroma provides persistent storage, metadata filtering, and document-level abstractions out of the box, while maintaining a lightweight footprint suitable for local development and small-to-medium deployments. <span style="color: #74B72E;">Additionally, it integrates natively with LangChain, enabling seamless management of embeddings, chunk metadata, source attribution, and retriever configuration. This tight integration is particularly important for medical Retrieval-Augmented Generation (RAG) systems, where traceability, reproducibility, and source transparency are essential. Furthermore, Chroma’s persistence model allows the vector store to be reused across application restarts, facilitating iterative experimentation and reliable deployment within Streamlit-based applications.</span>

In summary, **ChromaDB** was selected as the vector database for this medical RAG chatbot due to its strong alignment with the project’s core requirements: reliability, transparency, rapid iteration, and medical-domain traceability. While FAISS excels as a low-level Approximate Nearest Neighbor (ANN) engine, its lack of native persistence and metadata support introduces unnecessary engineering complexity for a medical chatbot that must surface source documents, manage document lifecycles, and integrate seamlessly with a user-facing interface. However, migration to enterprise-grade vector databases such as Pinecone or Milvus remains a viable future pathway as the system evolves toward larger-scale or distributed deployments.

<hr />

</div>

**<span style="color: #A1AEB1;">Large Language Model</span>**

<div style="text-align: justify;">

As aforementioned, OpenAI’s services were selected for both embedding generation and Large Language Model (LLM) and it is the core reasoning component of a Retrieval-Augmented Generation (RAG) system, particularly in medical applications where factual accuracy, contextual grounding, and controlled response behavior are critical. Therefore, the comparative analysis concentrates on three large language models within the OpenAI ecosystem including GPT-4o, GPT-4o-mini, and GPT-3.5-turbo, and I will evaluate their relative effectiveness and suitability for downstream retrieval-augmented generation tasks.

**GPT-4o** represents OpenAI’s most capable general-purpose model, offering state-of-the-art reasoning performance, strong instruction adherence, and superior contextual understanding. In medical Retrieval-Augmented Generation (RAG) scenarios, GPT-4o demonstrates excellent ability to synthesize retrieved clinical evidence, interpret nuanced medical terminology, and generate structured, well-grounded responses. Moreover, it performs particularly well when handling complex or ambiguous medical queries that require multi-step reasoning, cross-referencing multiple retrieved chunks, or summarizing clinical guidelines. Its responses exhibit lower hallucination rates when paired with high-quality retrieval and explicit grounding instructions.

**GPT-4o-mini** is optimized for cost efficiency and low latency while preserving a substantial portion of GPT-4o’s reasoning capability. In the context of a medical Retrieval-Augmented Generation (RAG) chatbot, GPT-4o-mini demonstrates strong performance in answering common medical questions, explaining symptoms, and summarizing retrieved medical documents with high coherence. While it may exhibit slightly weaker multi-hop reasoning than GPT-4o in highly complex scenarios, GPT-4o-mini remains highly reliable for the majority of user-facing medical queries. Its faster response times make it particularly suitable for interactive Streamlit applications where user experience and responsiveness are important.

**GPT-3.5-Turbo** has historically served as a baseline model for conversational Artificial Intelligence (AI) due to its low cost and fast inference. However, in medical Retrieval-Augmented Generation (RAG) settings, it exhibits notable weaknesses in reasoning accuracy, hallucination control, and instruction adherence when compared to GPT-4o-class models. While GPT-3.5-Turbo can handle simple factual queries or lightweight summarization tasks, its responses are more sensitive to prompt phrasing and retrieval noise. This increases the risk of generating speculative or weakly grounded answers in safety-sensitive medical contexts.

<span style="color: #74B72E;">

Following a structured evaluation aligned with the project’s medical Retrieval-Augmented Generation (RAG) requirements, GPT-4o-mini was selected as the primary Large Language Model for this chatbot. It provides the most balanced combination of reasoning reliability, latency, and cost efficiency for interactive medical question answering, particularly when paired with a robust retrieval layer and explicit grounding instructions.

GPT-4o remains the preferred upgrade path for scenarios requiring advanced clinical reasoning, multi-document synthesis, or safety-critical decision support, where maximum accuracy outweighs latency and cost considerations. In contrast, GPT-3.5-Turbo, while economical, does not meet the reliability and safety standards required for medical-domain applications and is therefore unsuitable as the default model.

In summary, this tiered model selection strategy mirrors the vector database decision process: prioritizing practical reliability and developer productivity (GPT-4o-mini) while retaining a clear pathway to higher-capability infrastructure (GPT-4o) as system complexity and clinical rigor increase.

</span>

</div>
