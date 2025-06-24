
import pandas as pd
import time
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
import os
from tqdm import tqdm
import json
import ast


def create_relevance_chain():
    """Create a LangChain chain for determining paper relevance"""

    # Create DeepSeek LLM instance
    llm = ChatOpenAI(
        api_key="your-token",  # Use environment variable
        base_url="https://api.deepseek.com/v1",  # Correct base URL with version
        model_name="deepseek-chat",  # DeepSeek model name
        temperature=0.1,  # Low temperature for more deterministic answers
        max_tokens=2048  # Limit output length
    )

    # Create prompt template
    template = """
    You are a specialized medical assistant focused on radiology report generation. You will be given:

    1. A generated caption for a radiology image
    2. Between 1-3 reference captions retrieved by a RAG system that may be similar to the current image
    
    Your task is to create a concise, accurate final caption of NO MORE THAN 80 WORDS.
    
    Important instructions:
    - First, internally analyze whether the reference captions describe similar findings to the generated caption
    - If references describe DIFFERENT scenarios (different pathologies, anatomical locations, or key findings), rely ONLY on the generated caption
    - If references describe SIMILAR findings, incorporate relevant details while maintaining the core observations from the generated caption
    - Use proper medical terminology and structure
    - Do not explain your reasoning or similarity assessment
    - Provide ONLY the final caption in your response
    - Ensure the final caption is 80 words or less
    
    Example input:
    Generated Caption: "CT scan shows a 3 cm hyperdense lesion in the right lobe of the liver with peripheral enhancement. No lymphadenopathy."
    
    Reference Caption 1: "Contrast-enhanced CT abdomen demonstrates a 3.2 cm hypervascular lesion in segment VI of the liver with peripheral enhancement in arterial phase and washout in venous phase, suspicious for hepatocellular carcinoma. No regional lymphadenopathy or ascites."
    
    Reference Caption 2: "MRI brain with contrast reveals a 2.5 cm extra-axial mass in the right frontal region with dural tail sign, consistent with meningioma. Minimal mass effect on adjacent brain parenchyma."
    
    Example output:
    Contrast-enhanced CT abdomen demonstrates a 3 cm hypervascular lesion in the right lobe of liver (segment VI) with peripheral enhancement in arterial phase and washout in venous phase, suspicious for hepatocellular carcinoma. No regional lymphadenopathy or ascites identified.
    
    
    Here is generated caption:{caption}
    
    References: {reference}

"""

    prompt = ChatPromptTemplate.from_template(template)
    # Create chain
    chain = prompt | llm

    return chain


if __name__ == "__main__":
    chain = create_relevance_chain()

    # cluster_rag_results_df = pd.read_csv("cluster_rag_results_testset.csv")
    cluster_rag_results_df = pd.read_csv("/home/jiawei/pyproject/ImageCLEFcompetition/test_results_generation/results/cluster_rag_results_v2.csv")

    rag_count = 0
    all_image_id = []
    all_result = []
    failed_ids = []  # List to store failed IDs

    for index, row in tqdm(cluster_rag_results_df.iterrows(),
                           total=len(cluster_rag_results_df),
                           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'):
        image_id = row["ID"]
        all_image_id.append(image_id)
        caption = row['Caption']
        similar_docs_str = row['similar_docs']
        similar_docs = ast.literal_eval(similar_docs_str)

        result = ""
        if similar_docs:
            rag_count += 1
            try:
                # Uncomment and use your chain invocation code here
                reference = ""
                for idx, doc in enumerate(similar_docs):
                    reference += f"{idx + 1}: {doc['caption']}\n"
                result = chain.invoke({"caption": caption, "reference": reference}).content
                time.sleep(0.5)
            except Exception as e:
                # Log the error
                print(f"Error processing ID {image_id}: {str(e)}")
                # Add the failed ID to the list
                failed_ids.append(image_id)
                # Continue with original caption
                result = caption
        else:
            result = caption

        all_result.append(result)

    # Save the results to CSV
    results_df = pd.DataFrame({
        'ID': all_image_id,
        'Generated_Caption': all_result
    }).to_csv("deepseek_rag_valid.csv", index=False, encoding="utf-8")

    # Save failed IDs to a text file
    if failed_ids:
        with open("failed_ids.txt", "w") as f:
            for failed_id in failed_ids:
                f.write(f"{failed_id}\n")
        print(f"Saved {len(failed_ids)} failed IDs to failed_ids.txt")