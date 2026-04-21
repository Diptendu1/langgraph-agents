from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import base64
from pathlib import Path
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.documents import Document

class Utility:

    def get_vector_store(self):
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key="AIzaSyChOUEIJfJ3CdoK4xxrRW_ecNi6ZcZlctI",
            task_type="retrieval_document",
            output_dimensionality=768
        )

        vectorstore = Chroma(
            collection_name="multimodal_index",
            embedding_function=embeddings,
            persist_directory="./chroma_db"
        )
        return vectorstore

    def handle_text_file(self, file_path):
        loader = TextLoader(file_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        for chunk in chunks:
            chunk.metadata.update({"file_type": "text", "source_file": file_path})

        self.get_vector_store().add_documents(chunks)
        return len(chunks)

    def index_pdf(self,file_path):
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        for chunk in chunks:
            chunk.metadata.update({"file_type": "pdf", "source_file": file_path})

        self.get_vector_store().add_documents(chunks)
        return len(chunks)

    def get_vision_model(self):
        vision_model = genai.GenerativeModel("gemini-2.0-flash")
        return vision_model

    def describe_image(self,image_path):
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        suffix = Path(image_path).suffix.lower()
        mime_map = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".webp": "image/webp"
        }
        mime_type = mime_map.get(suffix, "image/jpeg")

        response = self.get_vision_model().generate_content([
            {"mime_type": mime_type, "data": image_data},
            """Describe this image in detail for search indexing. Include:
            - What is shown (objects, people, scenes, activities)
            - Any visible text or labels (read them exactly)
            - Colors, layout, and composition if relevant
            - Any charts, graphs, or data shown (describe the values)
            - Context clues about the purpose or setting
            Be specific and comprehensive."""
        ])
        return response.text

    def index_image(self,image_path: str) -> int:
        description = self.describe_image(image_path)

        doc = Document(
            page_content=description,
            metadata={
                "file_type": "image",
                "source_file": image_path,
                "content_type": "image_description"
            }
        )
        self.get_vector_store().add_documents([doc])
        return 1

    def transcribe_audio(self, audio_path):
        with open(audio_path, "rb") as f:
            audio_data = base64.b64encode(f.read()).decode("utf-8")

        suffix = Path(audio_path).suffix.lower()
        mime_map = {
            ".mp3": "audio/mp3", ".wav": "audio/wav",
            ".m4a": "audio/mp4", ".ogg": "audio/ogg"
        }
        mime_type = mime_map.get(suffix, "audio/mp3")

        response = self.get_vision_model().generate_content([
            {"mime_type": mime_type, "data": audio_data},
            """Process this audio for search indexing:
            [TRANSCRIPT] Transcribe the spoken content verbatim.
            [SUMMARY] Summarize key topics, decisions, action items, and named entities.
            [SPEAKERS] Identify speakers if distinguishable."""
        ])
        return response.text

    def index_audio(self,audio_path):
        content = self.transcribe_audio(audio_path)

        if len(content) > 800:
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
            chunks = splitter.create_documents(
                [content],
                metadatas=[{"file_type": "audio", "source_file": audio_path}]
            )
            self.get_vector_store().add_documents(chunks)
            return len(chunks)

        doc = Document(
            page_content=content,
            metadata={"file_type": "audio", "source_file": audio_path}
        )
        self.get_vector_store().add_documents([doc])
        return 1

    def index_video(self,video_path):
        file_size = Path(video_path).stat().st_size

        if file_size < 20 * 1024 * 1024:
            with open(video_path, "rb") as f:
                video_data = base64.b64encode(f.read()).decode("utf-8")

            response = self.get_vision_model().generate_content([
                {"mime_type": "video/mp4", "data": video_data},
                """Analyze this video for search indexing:
                1. Main topic and purpose
                2. Visual content — scenes, objects, people, on-screen text
                3. Spoken content — transcribe or summarize
                4. Key moments with approximate timestamps
                5. Any brands, products, or entities mentioned"""
            ])
            content = response.text
        else:
            content = f"[Large video — manual segmentation required]: {video_path}"

        doc = Document(
            page_content=content,
            metadata={"file_type": "video", "source_file": video_path}
        )
        self.get_vector_store().add_documents([doc])
        return 1

    def search(self, query: str, k: int = 5, file_type: str = None) -> list[dict]:
        """
        Query the unified index.

        Args:
            query: Natural language query
            k: Number of results
            file_type: Optional filter — "text", "pdf", "image", "audio", "video"
        """
        where_filter = {"file_type": {"$eq": file_type}} if file_type else None

        # Use query-optimized embeddings for retrieval
        query_embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=os.environ["GOOGLE_API_KEY"],
            task_type="retrieval_query",
            output_dimensionality=768
        )
        query_store = Chroma(
            collection_name="multimodal_index",
            embedding_function=query_embeddings,
            persist_directory="./chroma_db"
        )

        results = query_store.similarity_search_with_score(query, k=k, filter=where_filter)

        return [
            {
                "content": doc.page_content[:400],
                "source_file": doc.metadata.get("source_file"),
                "file_type": doc.metadata.get("file_type"),
                "score": round(1 - score, 4)
            }
            for doc, score in results
        ]
