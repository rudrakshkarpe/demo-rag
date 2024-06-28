import imp  # pylint: disable=deprecated-module
import logging
import os
import sys
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai  # Ensure this is imported after the hook registration
from dotenv import load_dotenv
import PyPDF2
import io

# Import the generative AI import hook as defined in your Colab script
class _GenerativeAIImportHook:
    """Enables the PaLM and Gemini API clients libraries to be customized upon import."""

    def find_module(self, fullname, path=None):
        if fullname != 'google.generativeai':
            return None
        self.module_info = imp.find_module(
            fullname.split('.')[-1], list(path) if path else None
        )
        return self

    def load_module(self, fullname):
        """Loads google.generativeai normally and runs pre-initialization code."""
        previously_loaded = fullname in sys.modules
        generativeai_module = imp.load_module(fullname, *self.module_info)

        if not previously_loaded:
            try:
                import functools
                import json
                from google.colab import output
                from google.colab.html import _background_server
                import portpicker
                import tornado.web

                def fetch(request):
                    path = request.path
                    method = request.method
                    headers = json.dumps(dict(request.headers))
                    body = repr(request.body.decode('utf-8')) if request.body else 'null'
                    return output.eval_js("""
                      (async () => {{
                        // The User-Agent header causes CORS errors in Firefox.
                        const headers = {headers};
                        delete headers["User-Agent"];
                        const response = await fetch(new URL('{path}', 'https://generativelanguage.googleapis.com'), {{
                                    method: '{method}',
                                    body: {body},
                                    headers,
                                  }});
                        const json = await response.json();
                        return json;
                      }})()
                    """.format(path=path, method=method, headers=headers, body=body))

                class _Redirector(tornado.web.RequestHandler):
                    """Redirects API requests to the browser."""

                    async def get(self):
                        await self._handle_request()

                    async def post(self):
                        await self._handle_request()

                    async def _handle_request(self):
                        try:
                            result = fetch(self.request)
                            if isinstance(result, dict) and 'error' in result:
                                self.set_status(int(result['error']['code']))
                                self.write(result['error']['message'])
                                return
                            self.write(json.dumps(result))
                        except Exception as e:
                            self.set_status(500)
                            self.write(str(e))

                class _Proxy(_background_server._BackgroundServer):  # pylint: disable=protected-access
                    """Background server that intercepts API requests and then proxies the requests via the browser."""

                    def __init__(self):
                        app = tornado.web.Application([
                            (r'.*', _Redirector),
                        ])
                        super().__init__(app)

                    def create(self, port):
                        if self._server_thread is None:
                            self.start(port=port)

                port = portpicker.pick_unused_port()

                @functools.cache
                def start():
                    p = _Proxy()
                    p.create(port=port)
                    return p

                start()
                orig_configure = generativeai_module.configure
                generativeai_module.configure = functools.partial(
                    orig_configure,
                    transport='rest',
                    client_options={'api_endpoint': f'http://localhost:{port}'},
                )
            except:
                logging.exception('Error customizing google.generativeai.')
                os.environ['COLAB_GENERATIVEAI_IMPORT_HOOK_EXCEPTION'] = '1'

        return generativeai_module

def _register_hook():
    sys.meta_path = [_GenerativeAIImportHook()] + sys.meta_path

_register_hook()

# Load environment variables from .env file
load_dotenv("./example.env")
google_api_key = os.getenv("GOOGLE_API_KEY")

st.title("Hi there 👋")  # Updated title

# Configure generativeai module after loading API key
genai.configure(api_key=google_api_key)

# Check if the API key is available
if google_api_key is None:
    st.warning("API key not found. Please set the google_api_key environment variable.")
    st.stop()

# File Upload with user-defined name
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    st.text("PDF File Uploaded Successfully!")

    # PDF Processing (using PyPDF2 directly)
    pdf_data = uploaded_file.read()
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_data))
    pdf_pages = pdf_reader.pages

    # Create Context
    context = "\n\n".join(page.extract_text() for page in pdf_pages)

    # Split Texts
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
    texts = text_splitter.split_text(context)

    # Chroma Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_index = Chroma.from_texts(texts, embeddings).as_retriever()

    # Get User Question
    user_question = st.text_input("Ask a Question:")

    if st.button("Get Answer"):
        if user_question:
            # Get Relevant Documents
            docs = vector_index.get_relevant_documents(user_question)

            # Define Prompt Template
            prompt_template = """
            Answer the question as detailed as possible from the provided context,
            make sure to provide all the details, if the answer is not in
            provided context just say, "answer is not available in the context",
            don't provide the wrong answer\n\n
            Context:\n {context}?\n
            Question: \n{question}\n
            Answer:
            """

            # Create Prompt
            prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

            # Load QA Chain
            model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, api_key=google_api_key)
            chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

            # Get Response
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

            # Display Answer
            st.subheader("Answer:")
            st.write(response['output_text'])

        else:
            st.warning("Please enter a question.")
