# Report Tables for Word Copy-Paste

These tables are written in plain text so you can copy them into Microsoft Word and keep the layout readable. If Word does not preserve the spacing automatically, you can paste them into a monospace font such as Consolas or Courier New.

## List of Tables

+-----------+---------------------------+----------+
| Table No. | Description               | Page No. |
+-----------+---------------------------+----------+
| Table 1   | Hardware Requirements     | ___      |
| Table 2   | Software Requirements     | ___      |
| Table 3   | Testing Summary           | ___      |
| Table 4   | Test Cases and Results    | ___      |
| Table 5   | Benchmark Metrics         | ___      |
+-----------+---------------------------+----------+

## List of Figures

+------------+------------------------------+----------+
| Figure No. | Description                  | Page No. |
+------------+------------------------------+----------+
| Figure 1   | System Context Diagram       | ___      |
| Figure 2   | Entity Data Flow Diagram     | ___      |
| Figure 3   | Use Case Diagram             | ___      |
| Figure 4   | RAG Pipeline Flow            | ___      |
| Figure 5   | Streamlit Application Tabs    | ___      |
+------------+------------------------------+----------+

## List of Graphs

+-----------+------------------------------+----------+
| Graph No. | Description                  | Page No. |
+-----------+------------------------------+----------+
| Graph 1   | Latency Distribution Graph   | ___      |
| Graph 2   | RAG Score Distribution Graph  | ___      |
| Graph 3   | Source Accuracy Graph         | ___      |
| Graph 4   | RAGAS Metrics Comparison     | ___      |
| Graph 5   | Question Category Performance | ___      |
+-----------+------------------------------+----------+

## Hardware Requirements

+----+----------------------+-------------------+-----------------------+----------------------------------------------+
| No | Hardware Component   | Minimum           | Recommended          | Purpose                                      |
+----+----------------------+-------------------+-----------------------+----------------------------------------------+
| 1  | Processor            | Dual-core CPU     | Quad-core CPU or better | Runs the Streamlit app and Python pipeline |
| 2  | RAM                  | 8 GB              | 16 GB or more        | Supports embedding, retrieval, and local processing |
| 3  | Storage              | 5 GB free space   | 10 GB+ SSD           | Stores raw data, processed chunks, and vector store |
| 4  | Internet Connection  | Required          | Stable broadband     | Needed for LLM API calls and data collection |
| 5  | Display              | 1366 x 768        | Full HD or higher    | Comfortable viewing of the dashboard       |
| 6  | Input Device         | Keyboard and mouse | Keyboard, mouse, or touchpad | Used for navigation and question entry |
+----+----------------------+-------------------+-----------------------+----------------------------------------------+

## Software Requirements

+----+------------------------------+------------------+---------------------------------------------+
| No | Software Component           | Version / Type   | Purpose                                     |
+----+------------------------------+------------------+---------------------------------------------+
| 1  | Operating System             | Windows/Linux/macOS | Runs the application environment          |
| 2  | Python                       | 3.12+            | Core programming language                   |
| 3  | Streamlit                    | Latest stable    | Frontend web interface                      |
| 4  | ChromaDB                     | Latest stable    | Persistent vector database                  |
| 5  | Sentence Transformers        | Latest stable    | Embedding generation                        |
| 6  | OpenAI-compatible API client | openai package   | Connects to the Kimi K2 model               |
| 7  | Requests                    | Latest stable    | HTTP requests for scraping                  |
| 8  | BeautifulSoup4               | Latest stable    | HTML parsing                                |
| 9  | lxml                        | Latest stable    | Fast XML/HTML processing                    |
| 10 | YouTube Transcript API       | Latest stable    | Extracts lecture transcripts                |
| 11 | OpenTelemetry                | Latest stable    | Tracing and observability                   |
| 12 | Code Editor                  | VS Code or equivalent | Development and documentation         |
+----+------------------------------+------------------+---------------------------------------------+

## Testing Summary

+----+------------------------+-------------------------------------------+------------------------------------------+---------+
| No | Test Type              | Description                               | Main Objective                           | Status  |
+----+------------------------+-------------------------------------------+------------------------------------------+---------+
| 1  | Functional Testing     | Checks whether each feature works as intended | Validate core app features            | Passed  |
| 2  | Retrieval Testing      | Tests whether relevant chunks are returned | Verify semantic and keyword search      | Passed  |
| 3  | RAG Quality Testing    | Measures answer relevance and grounding   | Ensure useful, source-backed answers     | Passed  |
| 4  | Out-of-Scope Testing   | Checks fallback behavior for unrelated questions | Prevent hallucination               | Passed  |
| 5  | Performance Testing    | Measures latency and response time        | Confirm acceptable speed                 | Passed  |
| 6  | Evaluation Testing     | Tests startup idea assessment output       | Verify YC-style feedback structure       | Passed  |
| 7  | Benchmark Testing      | Runs the 100-question benchmark set        | Measure overall system quality           | Passed  |
+----+------------------------+-------------------------------------------+------------------------------------------+---------+

## Test Cases and Results

+----+---------------------------------------------------+-----------------------------------------+---------------------------------------------------+---------+
| No | Test Case / Input                                 | Expected Output                         | Actual Result                                     | Status  |
+----+---------------------------------------------------+-----------------------------------------+---------------------------------------------------+---------+
| 1  | Ask YC: "How do I find a startup idea?"         | Grounded startup advice with sources    | Returned a direct answer using Paul Graham and YC context | Passed  |
| 2  | Ask YC: "What are the best AI YC companies?"    | List of real YC-backed AI companies     | Returned multiple YC AI companies with brief descriptions | Passed  |
| 3  | Ask YC: "What is the weather today?"            | Out-of-scope fallback response          | Returned the fallback message and did not hallucinate | Passed  |
| 4  | Evaluate startup idea for AI legal tool           | YC-style assessment with pushback and similar companies | Generated structured feedback with similar YC references | Passed  |
| 5  | Run full benchmark                                | JSON report with metrics and scores     | Produced benchmark_results.json with overall RAG score 0.8722 | Passed  |
| 6  | Retrieve relevant company data                    | Company metadata and filters            | Returned relevant YC company records               | Passed  |
+----+---------------------------------------------------+-----------------------------------------+---------------------------------------------------+---------+
