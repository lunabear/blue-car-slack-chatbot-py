from azure.search.documents.indexes._generated.models import SemanticSettings, SemanticConfiguration, PrioritizedFields, \
    SemanticField
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_core.prompts import PromptTemplate

from constants import AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL, \
    AZURE_OPENAI_API_VERSION, AZURE_SEARCH_INDEX_NAME, AZURE_SEARCH_ADMIN_KEY, AZURE_SEARCH_SERVICE_ENDPOINT
from utils.langchain_util import get_chat_azure_openai_client, LLMNames
from utils.s3_utils import get_object_contents


def query_to_llm(query: str) -> str:
    llm = get_chat_azure_openai_client(
        api_key=AZURE_OPENAI_API_KEY,
        endpoint=AZURE_OPENAI_ENDPOINT,
        deployment_name=LLMNames.gpt_35_turbo_16k,  # gpt-4 32k
        max_tokens=4096,
    )


    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=get_vector_store().as_retriever(),
        return_source_documents=False,
        # chain_type_kwargs={
        #     "prompt": PromptTemplate(
        #         template=template,
        #         input_variables=["summaries", "question"],
        #     ),
        # },

    )

    return qa.invoke(query)


def get_vector_store():
    embeddings: OpenAIEmbeddings = AzureOpenAIEmbeddings(
        azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL,
        model=AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL,
        chunk_size=1,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        openai_api_type='azure',
        api_version=AZURE_OPENAI_API_VERSION,
    )

    # Create an index in Azure Search
    vector_store: AzureSearch = AzureSearch(
        azure_search_endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
        azure_search_key=AZURE_SEARCH_ADMIN_KEY,
        index_name=AZURE_SEARCH_INDEX_NAME,
        embedding_function=embeddings.embed_query,
        semantic_configuration_name='config',
        semantic_settings=SemanticSettings(
            default_configuration='config',
            configurations=[
                SemanticConfiguration(
                    name='config',
                    prioritized_fields=PrioritizedFields(
                        title_field=SemanticField(field_name='content'),
                        prioritized_content_fields=[SemanticField(field_name='content')],
                        prioritized_keywords_fields=[SemanticField(field_name='metadata')]
                    ))
            ])
    )

    return vector_store


def insert_to_vector_store(s3_key: str):
    vector_store = get_vector_store()

    print('Create index')

    binary = get_object_contents(object_key=s3_key)
    with open('/tmp/test.pdf', 'wb') as f:
        f.write(binary)
    loader = PyPDFLoader('/tmp/test.pdf')
    document = loader.load()
    # print(document[2].page_content[:20000])

    import tiktoken  # !pip install tiktoken

    tokenizer = tiktoken.get_encoding('cl100k_base')  # GPT Turobo 3.5 토큰 방식

    # 코드로 토큰을 분리하는 방법
    # create the length function
    def tiktoken_len(text):
        tokens = tokenizer.encode(
            text,
            disallowed_special=()
        )
        return len(tokens)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=20,
        length_function=tiktoken_len,
        separators=["\n\n", "\n", " ", ""]
    )
    texts = text_splitter.split_documents(document)
    vector_store.add_documents(documents=texts)


if __name__ == '__main__':
    s3_key = 'gs_insa_test.pdf'
    insert_to_vector_store(s3_key)