# LLM Ad Placement Testing

## Project Purpose
This project provides a robust data pipeline and evaluation framework for testing the placement of advertisements within LLM-generated text responses. It simulates an advertising engine by matching synthetic ads to user queries from the LMSYS Chatbot Arena dataset, injecting them into responses, and using a strong LLM as a judge to evaluate the naturalness and quality of the placement.

## High-Level Architecture
The project follows a modular, sequential pipeline:
1. **`src/ad_generation`**: Data extraction and ad generation.
2. **`src/ad_indexing`**: Semantic indexing of ads using FAISS and BGE-M3.
3. **`src/ad_integration`**: Matching queries to ads and placing them in text via multiple strategies.
4. **`src/ad_evaluation`**: LLM-as-a-judge evaluation of ad placements.
5. **`src/common`**: Shared configuration and centralized `vLLM` generator classes.

## Important Development Constraints
- Use the virtual environment `.lmarena-env` for dependencies.
- The project requires GPU access for vLLM (fp16, fp8) and sentence-transformers inference.
- Shared configurations are maintained centrally in `src/common/shared_config.py`.
- Shared LLM behaviors (token truncation, chat formatting) are maintained in `src/common/shared_llm.py`.

## Documentation and Rules
- **Detailed Architecture**: Consult `docs/architecture.md` before making cross-module architectural changes.
- **Engineering Rules**: Consult `.agents/rules/python.md` for Python coding standards, SOLID application, and design pattern guidelines.
