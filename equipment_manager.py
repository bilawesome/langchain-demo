# more granural showcase of the whole process
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader

import os
os.environ["OPENAI_API_KEY"] = "your-api-key"


# reader = PdfReader("2023_GPT4All_Technical_Report.pdf")
reader = PdfReader("EquipmentManagerRolesAndResponsibilities.pdf")

# cut into chunks
raw_text = ''
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if not text:
        continue

    raw_text += text

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

texts = text_splitter.split_text(raw_text)
# didnt need to do any processing except for splitting the data.
# print('texts', texts[:50])

embeddings = OpenAIEmbeddings()
# Pinecone, Chroma

# input('..')

docsearch = FAISS.from_texts(texts, embeddings)


loader = CSVLoader(file_path="Equipment Tracking 2023 - Masterlist.csv")
# loader = CSVLoader(file_path="Copy of Equipment Tracking 2023 - Masterlist.csv")
documents = loader.load()
# print('documents', documents)
# didnt use textsplitter because the csv file is too short

db_csv = FAISS.from_documents(documents, embeddings)
db_pdf = FAISS.from_texts(texts, embeddings)

def retrieve_info(query):
    similarity_response_csv = db_csv.similarity_search(query)
    similarity_response_pdf = db_pdf.similarity_search(query)

    page_contents_array_csv = [doc.page_content for doc in similarity_response_csv]
    # has a lot of fluff in the response so need to return just the content

    page_contents_array_pdf = [doc.page_content for doc in similarity_response_pdf]

    return page_contents_array_pdf, page_contents_array_csv
    # return page_contents_array_csv + page_contents_array_pdf

llm = ChatOpenAI(model="gpt-4")
# you can specify what model to use

# you can also define the template
template = """
You are a virtual Equipment Manager, ready to assist you with equipment-related needs.
Whether employees are looking for information on available tools, requesting a repair, booking equipment, or needing guidance on proper usage, you're there to help.
Please choose from the following options, or type your specific question
You are an equipment manager chatbot for a software advancement consultancy firm.
Your roles and responsibilities are described in the role description.

I will share an employee's query with you and you will give me the best answer that I should
send to this employee based on the company equipment masterlist. The equipment masterlist is an
up-to-date inventory of all company-owned equipment, including laptops, peripherals, software licenses,
and any other related items.

Here's a description of the columns in the equipment masterlist:

Type: The type of equipment, which can be a "Desktop," "Laptop," "Phone," or "iPad."
Equipment ID: A unique identifier for each piece of equipment.
Date Issued: The date when the equipment was issued.
Date Returned: The date when the equipment was returned (blank for items that have not been returned).
Duration: The duration for which the equipment was used.
Previous Owner: The name of the previous owner of the equipment.
Issued To: The name of the person to whom the equipment was issued.
Date Submitted AA Form: The date when the AA (Asset Acquisition) Form was submitted.
Link to Signed AA Form: A link to the signed AA Form.
Date Enrolled in MDM: The date when the equipment was enrolled in Mobile Device Management (MDM).
Status: The current status of the equipment (e.g., "Issued", "For Selling").
Device Password: The password associated with the device (if applicable).
Status Report Link: A link to a status report.
FileVault Encryption: Whether FileVault encryption is enabled for the device (Yes/No).
Model Name: The name of the equipment's model (e.g., "Macbook Pro Retina (13 Inch)", "iPhone SE (Product Red)").
Year: The year of the equipment's model.
Model Number: The model number of the equipment.
Serial Number: The unique serial number of the equipment.
Storage: The storage capacity of the equipment (e.g., "256 GB", "64 GB").
RAM: The amount of RAM in the equipment (e.g., "8 GB", "2 GB").
Processor: The processor details of the equipment (e.g., "Dual-Core Intel Core i5", "A15 Bionic").
Company Name of Supplier: The name of the company that supplied the equipment.
Acquisition Cost: The cost of acquiring the equipment.
Warranty Start: The start date of the warranty for the equipment.
Warranty End: The end date of the warranty for the equipment.

Employees will ask rhetorical questions pertaining to the equipment masterlist and you should be able to provide an accurate answer as equipment manager.

query: {query}
==================
role description: {role_description}
==================
equipment masterlist: {equipment_masterlist}
"""

prompt = PromptTemplate(
    input_variables=["query", "role_description", "equipment_masterlist"],
    template=template
) 

chain = LLMChain(llm=llm, prompt=prompt)

# for this process compared to app.py, we need to define the LLM, the prompt, and the embedding, before we
# can run the chain.
# this is different from app.py where it assumes the default for the LLM, the prompt being just a simple Q&A.
# the embedding it assumes tWho he input variable of input_documents. But overall its the same.
def generate_response(query):
    pdf, csv = retrieve_info(query)
    response = chain.run(query=query, role_description=pdf, equipment_masterlist=csv)
    # runs based on the input you defined in the prompt variable
    return response


if __name__ == "__main__":
    while True:
        print('================================================')
        query = input("Query: ")
        response = generate_response(query)
        print('Virtual Equipment Manager:', response)
        print('\n')
